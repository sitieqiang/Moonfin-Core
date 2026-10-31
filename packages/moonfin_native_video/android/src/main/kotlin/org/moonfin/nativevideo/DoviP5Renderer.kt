package org.moonfin.nativevideo

import android.graphics.SurfaceTexture
import android.media.MediaFormat
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import androidx.media3.common.C
import androidx.media3.common.ColorInfo
import androidx.media3.common.Format
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.video.VideoFrameMetadataListener
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentSkipListMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The RPUs of a profile 5 track by presentation time. The extractor adds to it
 * ahead of playback and the renderer reads from it, on their own threads.
 */
internal class DoviP5RpuStore {
    private val rpus = ConcurrentSkipListMap<Long, ByteArray>()

    fun add(timeUs: Long, data: ByteArray, offset: Int, length: Int) {
        rpus[timeUs] = data.copyOfRange(offset, offset + length)
        while (rpus.size > MAX_RPUS) {
            rpus.pollFirstEntry()
        }
    }

    fun floorEntry(timeUs: Long): Map.Entry<Long, ByteArray>? = rpus.floorEntry(timeUs)

    private companion object {
        // A few minutes at 60fps, which outlasts the player's buffer.
        const val MAX_RPUS = 12_000
    }
}

/**
 * Draws Dolby Vision profile 5 on a display without Dolby Vision, for devices
 * whose decoder can't map it themselves.
 *
 * The track decodes as plain HEVC into a [SurfaceTexture], and each frame is
 * reshaped on the GPU with the RPU matching its presentation time, then tone
 * mapped to SDR. The output is SDR because the only HDR surface the SHIELD
 * offers is FP16, and it can't sustain that at 4K60.
 *
 * The GL driver turns the decoded YUV into RGB before the shader sees it, since
 * the SHIELD has no GL_EXT_YUV_target. The shader inverts that matrix, but
 * anything the driver clipped on the way is gone, so very saturated highlights
 * can land a little off.
 */
@UnstableApi
internal class DoviP5Renderer(
    private val window: Surface,
    private val rpus: DoviP5RpuStore,
    private val onInputSurface: (Surface) -> Unit,
    private val onFailed: () -> Unit,
) : VideoFrameMetadataListener {

    private val thread = HandlerThread("DoviP5Renderer").apply { start() }
    private val handler = Handler(thread.looper)
    private val releaseToPts = ConcurrentSkipListMap<Long, Long>()

    private var display: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var context: EGLContext = EGL14.EGL_NO_CONTEXT
    private var surface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var texture: SurfaceTexture? = null
    private var input: Surface? = null

    private val mapping = FloatArray(MAPPING_LENGTH)
    private val pivots = FloatArray(27)
    private val heads = FloatArray(3 * MAX_PIECES * 4)
    private val mmr = FloatArray(3 * MAX_PIECES * 21)
    private val counts = IntArray(3)
    private val transform = FloatArray(16)
    private var haveMapping = false
    private var haveMatrices = false
    private var lastRpuTimeUs = Long.MIN_VALUE
    private var framesWithoutRpu = 0

    private var frames = 0
    private var drawNanos = 0L
    private var statsSince = SystemClock.elapsedRealtime()

    private var uReshape = -1
    private var uTransform = -1
    private var uCounts = -1
    private var uPivots = -1
    private var uHeads = -1
    private var uMmr = -1
    private var uNonlinear = -1
    private var uNonlinearOffset = -1
    private var uLms = -1

    init {
        handler.post {
            runCatching { setUp() }.onFailure {
                Log.e(TAG, "GL setup failed", it)
                onFailed()
            }
        }
    }

    override fun onVideoFrameAboutToBeRendered(
        presentationTimeUs: Long,
        releaseTimeNs: Long,
        format: Format,
        mediaFormat: MediaFormat?,
    ) {
        releaseToPts[releaseTimeNs] = presentationTimeUs
        while (releaseToPts.size > MAX_PENDING_FRAMES) {
            releaseToPts.pollFirstEntry()
        }
    }

    /**
     * Stops drawing and lets go of the window. Waits for that, since the
     * window's owner may destroy it as soon as this returns.
     */
    fun release() {
        val done = CountDownLatch(1)
        handler.post {
            runCatching { tearDown() }
            thread.quitSafely()
            done.countDown()
        }
        done.await(RELEASE_TIMEOUT_MS, TimeUnit.MILLISECONDS)
    }

    private fun tearDown() {
        texture?.setOnFrameAvailableListener(null)
        texture?.release()
        input?.release()
        texture = null
        input = null
        if (display != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(
                display,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_CONTEXT,
            )
            EGL14.eglDestroySurface(display, surface)
            EGL14.eglDestroyContext(display, context)
        }
    }

    private fun setUp() {
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val version = IntArray(2)
        check(EGL14.eglInitialize(display, version, 0, version, 1)) { "eglInitialize failed" }

        val config = chooseConfig() ?: error("no RGBA8888 EGL config")
        context = EGL14.eglCreateContext(
            display,
            config,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE),
            0,
        )
        check(context != EGL14.EGL_NO_CONTEXT) { "no ES3 context" }
        surface = EGL14.eglCreateWindowSurface(display, config, window, intArrayOf(EGL14.EGL_NONE), 0)
        check(surface != EGL14.EGL_NO_SURFACE) {
            "no window surface (0x${EGL14.eglGetError().toString(16)})"
        }
        check(EGL14.eglMakeCurrent(display, surface, surface, context)) { "eglMakeCurrent failed" }

        val program = buildProgram()
        GLES30.glUseProgram(program)
        uReshape = GLES30.glGetUniformLocation(program, "uReshape")
        uTransform = GLES30.glGetUniformLocation(program, "uTransform")
        uCounts = GLES30.glGetUniformLocation(program, "uCounts")
        uPivots = GLES30.glGetUniformLocation(program, "uPivots")
        uHeads = GLES30.glGetUniformLocation(program, "uHeads")
        uMmr = GLES30.glGetUniformLocation(program, "uMmr")
        uNonlinear = GLES30.glGetUniformLocation(program, "uNonlinear")
        uNonlinearOffset = GLES30.glGetUniformLocation(program, "uNonlinearOffset")
        uLms = GLES30.glGetUniformLocation(program, "uLms")
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uTexture"), 0)

        val quad = ByteBuffer.allocateDirect(QUAD.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(QUAD)
        quad.position(0)
        // A buffer object, since GL would only keep a pointer into the Java
        // buffer and that memory goes away once it is collected.
        val buffers = IntArray(1)
        GLES30.glGenBuffers(1, buffers, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, buffers[0])
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, QUAD.size * 4, quad, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, 0)

        val names = IntArray(1)
        GLES30.glGenTextures(1, names, 0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, names[0])
        for (filter in intArrayOf(GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_TEXTURE_MAG_FILTER)) {
            GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, filter, GLES30.GL_LINEAR)
        }
        val created = SurfaceTexture(names[0])
        created.setOnFrameAvailableListener({ draw() }, handler)
        texture = created
        val inputSurface = Surface(created)
        input = inputSurface
        Log.i(TAG, "ready on ${GLES30.glGetString(GLES30.GL_RENDERER)}")
        onInputSurface(inputSurface)
    }

    private fun chooseConfig(): EGLConfig? {
        val attributes = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(64)
        val count = IntArray(1)
        EGL14.eglChooseConfig(display, attributes, 0, configs, 0, configs.size, count, 0)
        // eglChooseConfig only treats the sizes as minimums, so the exact one
        // has to be picked out.
        return configs.take(count[0]).filterNotNull().firstOrNull { config ->
            configSize(config, EGL14.EGL_RED_SIZE) == 8 &&
                configSize(config, EGL14.EGL_GREEN_SIZE) == 8 &&
                configSize(config, EGL14.EGL_BLUE_SIZE) == 8 &&
                configSize(config, EGL14.EGL_ALPHA_SIZE) == 8
        }
    }

    private fun configSize(config: EGLConfig, attribute: Int): Int {
        val value = IntArray(1)
        EGL14.eglGetConfigAttrib(display, config, attribute, value, 0)
        return value[0]
    }

    private fun draw() {
        val current = texture ?: return
        val start = SystemClock.elapsedRealtimeNanos()
        current.updateTexImage()
        current.getTransformMatrix(transform)

        val releaseNs = current.timestamp
        releaseToPts.floorEntry(releaseNs + PTS_MATCH_WINDOW_NS)
            ?.takeIf { kotlin.math.abs(it.key - releaseNs) <= PTS_MATCH_WINDOW_NS }
            ?.let { loadRpu(it.value) }

        val reshaped = haveMapping && haveMatrices
        // Unmapped frames are green and purple, so the first few wait for an
        // RPU. A track that never yields one still has to show something.
        if (!reshaped && framesWithoutRpu++ < MAX_FRAMES_WITHOUT_RPU) return

        val width = IntArray(1)
        val height = IntArray(1)
        EGL14.eglQuerySurface(display, surface, EGL14.EGL_WIDTH, width, 0)
        EGL14.eglQuerySurface(display, surface, EGL14.EGL_HEIGHT, height, 0)
        GLES30.glViewport(0, 0, width[0], height[0])

        GLES30.glUniformMatrix4fv(uTransform, 1, false, transform, 0)
        GLES30.glUniform1i(uReshape, if (reshaped) 1 else 0)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        EGL14.eglSwapBuffers(display, surface)

        frames++
        drawNanos += SystemClock.elapsedRealtimeNanos() - start
        val elapsed = SystemClock.elapsedRealtime() - statsSince
        if (elapsed >= STATS_INTERVAL_MS) {
            Log.i(
                TAG,
                "fps=${"%.1f".format(frames * 1000f / elapsed)} " +
                    "draw=${"%.2f".format(drawNanos / 1e6 / frames)}ms " +
                    "output=${width[0]}x${height[0]} reshaped=$reshaped",
            )
            frames = 0
            drawNanos = 0
            statsSince += elapsed
        }
    }

    private fun loadRpu(pts: Long) {
        val entry = rpus.floorEntry(pts) ?: return
        if (entry.key == lastRpuTimeUs) return
        lastRpuTimeUs = entry.key

        val flags = DoviRpu.p5Mapping(entry.value, mapping)
        if (flags < 0) return
        // A frame reusing the previous mapping or matrices leaves them out, and
        // the uniforms already hold them.
        if (flags and 1 != 0) {
            for (c in 0 until 3) {
                val block = 21 + c * COMPONENT_BLOCK
                counts[c] = mapping[block].toInt()
                System.arraycopy(mapping, block + 1, pivots, c * 9, 9)
                System.arraycopy(mapping, block + 10, heads, c * MAX_PIECES * 4, MAX_PIECES * 4)
                System.arraycopy(mapping, block + 42, mmr, c * MAX_PIECES * 21, MAX_PIECES * 21)
            }
            GLES30.glUniform1iv(uCounts, 3, counts, 0)
            GLES30.glUniform1fv(uPivots, 27, pivots, 0)
            GLES30.glUniform4fv(uHeads, 3 * MAX_PIECES, heads, 0)
            GLES30.glUniform4fv(uMmr, mmr.size / 4, mmr, 0)
            haveMapping = true
        }
        if (flags and 2 != 0) {
            // The RPU's rows, which GL takes as columns.
            val m = mapping
            GLES30.glUniformMatrix3fv(
                uNonlinear,
                1,
                false,
                floatArrayOf(m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]),
                0,
            )
            GLES30.glUniform3f(uNonlinearOffset, m[9], m[10], m[11])
            GLES30.glUniformMatrix3fv(
                uLms,
                1,
                false,
                multiplyRows(LMS_TO_BT2020, FloatArray(9) { m[12 + it] }),
                0,
            )
            haveMatrices = true
        }
    }

    private fun buildProgram(): Int {
        val built = GLES30.glCreateProgram()
        GLES30.glAttachShader(built, compile(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER))
        GLES30.glAttachShader(built, compile(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER))
        GLES30.glBindAttribLocation(built, 0, "aPosition")
        GLES30.glLinkProgram(built)
        val status = IntArray(1)
        GLES30.glGetProgramiv(built, GLES30.GL_LINK_STATUS, status, 0)
        check(status[0] != 0) { "link failed: ${GLES30.glGetProgramInfoLog(built)}" }
        return built
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val status = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0)
        check(status[0] != 0) { "compile failed: ${GLES30.glGetShaderInfoLog(shader)}" }
        return shader
    }

    companion object {
        private const val TAG = "DoviP5"
        private const val MAX_PIECES = 8
        private const val COMPONENT_BLOCK = 210
        private const val MAPPING_LENGTH = 21 + 3 * COMPONENT_BLOCK
        private const val MAX_PENDING_FRAMES = 256
        private const val MAX_FRAMES_WITHOUT_RPU = 30
        private const val PTS_MATCH_WINDOW_NS = 4_000_000L
        private const val RELEASE_TIMEOUT_MS = 500L
        private const val STATS_INTERVAL_MS = 10_000L

        /**
         * Asks for a full range BT.709 conversion, which is the matrix the
         * shader inverts. The stream is full range, so a limited range matrix
         * would clip its blacks and whites.
         */
        val decoderColorInfo: ColorInfo = ColorInfo.Builder()
            .setColorSpace(C.COLOR_SPACE_BT709)
            .setColorRange(C.COLOR_RANGE_FULL)
            .setColorTransfer(C.COLOR_TRANSFER_SDR)
            .build()

        private val QUAD = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)

        // Dolby Vision output is BT.2020-referred HPE LMS.
        private val LMS_TO_BT2020 = floatArrayOf(
            3.06441879f, -2.16597676f, 0.10155818f,
            -0.65612108f, 1.78554118f, -0.12943749f,
            0.01736321f, -0.04725154f, 1.03004253f,
        )

        /** Multiplies two row major 3x3 matrices, returning column major for GL. */
        private fun multiplyRows(a: FloatArray, b: FloatArray): FloatArray {
            val result = FloatArray(9)
            for (row in 0 until 3) {
                for (column in 0 until 3) {
                    var sum = 0f
                    for (k in 0 until 3) {
                        sum += a[row * 3 + k] * b[k * 3 + column]
                    }
                    result[column * 3 + row] = sum
                }
            }
            return result
        }

        private val VERTEX_SHADER = """
            #version 300 es
            in vec2 aPosition;
            uniform mat4 uTransform;
            out vec2 vCoord;
            void main() {
                gl_Position = vec4(aPosition, 0.0, 1.0);
                vCoord = (uTransform * vec4(aPosition * 0.5 + 0.5, 0.0, 1.0)).xy;
            }
        """.trimIndent()

        private val FRAGMENT_SHADER = """
            #version 300 es
            #extension GL_OES_EGL_image_external_essl3 : require
            precision highp float;
            in vec2 vCoord;
            out vec4 fragColor;
            uniform samplerExternalOES uTexture;
            uniform int uReshape;
            uniform int uCounts[3];
            uniform float uPivots[27];
            uniform vec4 uHeads[24];
            uniform vec4 uMmr[126];
            uniform mat3 uNonlinear;
            uniform vec3 uNonlinearOffset;
            uniform mat3 uLms;

            const float M1 = 2610.0 / 16384.0;
            const float M2 = 2523.0 / 4096.0 * 128.0;
            const float C1 = 3424.0 / 4096.0;
            const float C2 = 2413.0 / 4096.0 * 32.0;
            const float C3 = 2392.0 / 4096.0 * 32.0;

            // Inverts the driver's full range BT.709 YUV to RGB, rows being Y,
            // Cb and Cr. See decoderColorInfo.
            const mat3 RGB_TO_YUV = mat3(
                0.2126, -0.114572, 0.5,
                0.7152, -0.385428, -0.454153,
                0.0722, 0.5, -0.045847);

            // Takes BT.2020 to BT.709.
            const mat3 BT2020_TO_BT709 = mat3(
                1.6605, -0.1246, -0.0182,
                -0.5876, 1.1329, -0.1006,
                -0.0728, -0.0083, 1.1187);

            vec3 pqToLinear(vec3 v) {
                vec3 p = pow(clamp(v, 0.0, 1.0), vec3(1.0 / M2));
                return pow(max(p - C1, 0.0) / (C2 - C3 * p), vec3(1.0 / M1));
            }

            float mmrCoef(int index) {
                return uMmr[index / 4][index % 4];
            }

            vec3 mmrCoefs(int index) {
                return vec3(mmrCoef(index), mmrCoef(index + 1), mmrCoef(index + 2));
            }

            float reshape(int c, vec3 s) {
                float x = s[c];
                int count = uCounts[c];
                int piece = 0;
                while (piece < count - 1 && x >= uPivots[c * 9 + piece + 1]) {
                    piece++;
                }
                vec4 head = uHeads[c * 8 + piece];
                if (head.x < 0.5) {
                    return clamp(head.y + x * (head.z + x * head.w), 0.0, 1.0);
                }
                vec3 cross1 = vec3(s.x * s.y, s.x * s.z, s.y * s.z);
                float cross2 = cross1.x * s.z;
                vec3 p = s;
                vec3 q = cross1;
                float r = cross2;
                float result = head.y;
                int order = int(head.z);
                for (int j = 0; j < order; j++) {
                    int base = c * 168 + piece * 21 + j * 7;
                    result += dot(mmrCoefs(base), p) + dot(mmrCoefs(base + 3), q)
                        + mmrCoef(base + 6) * r;
                    p *= s;
                    q *= cross1;
                    r *= cross2;
                }
                return clamp(result, 0.0, 1.0);
            }

            void main() {
                vec3 rgb = texture(uTexture, vCoord).rgb;
                if (uReshape == 0) {
                    fragColor = vec4(rgb, 1.0);
                    return;
                }
                vec3 ipt = clamp(RGB_TO_YUV * rgb + vec3(0.0, 0.5, 0.5), 0.0, 1.0);
                vec3 shaped = vec3(reshape(0, ipt), reshape(1, ipt), reshape(2, ipt));
                vec3 lms = pqToLinear(uNonlinear * (shaped - uNonlinearOffset));
                vec3 linear = max(uLms * lms, 0.0);
                // An extended Reinhard. PQ's 1.0 is 10000 nits, SDR white is
                // taken as 203, and 2030 nits is what reaches full white.
                vec3 bt709 = max(BT2020_TO_BT709 * linear, 0.0) * (10000.0 / 203.0);
                vec3 mapped = bt709 * (1.0 + bt709 / 100.0) / (1.0 + bt709);
                fragColor = vec4(pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
            }
        """.trimIndent()
    }
}
