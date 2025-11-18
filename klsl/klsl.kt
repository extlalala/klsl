package klsl

import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicReferenceArray
import java.util.stream.IntStream
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.time.measureTime

fun main() {
	val path = TODO("your output file path")
	val out = Files.newOutputStream(Path.of(path), StandardOpenOption.WRITE).buffered()
	out.use { test4(it) }
}

// tests from AI

fun test1(out: OutputStream) {
	val shader = klsl(DefaultFrontend(PPMBackend(out))) {
		mainImage {
			var U = (C + C - iResolution.xy) / iResolution.y
			U *= mat2(cos(iTime), sin(iTime), -sin(iTime), cos(iTime))
			U *= 5.0f
			val p = length(U) * vec2(cos(atan(U.y, U.x) * 7.0f), sin(atan(U.y, U.x) * 6.0f))
			O = vec4(sin(p.x + iTime) + sin(p.y * 2.0f), cos(p.x * p.y), sin(length(U) * 3.0f), 1f) * .5f + .5f
		}
	}
	val time = measureTime {
		shader.render(ivec2(2000, 2000), 0f)
	}
	println(time)
}

fun test2(out: OutputStream) {
	val shader = klsl(DefaultFrontend(PPMBackend(out))) {
		mainImage {
			var U = C / iResolution.xy - .5f
			val u = U
			U = vec2(U.x * (iResolution.x / iResolution.y), U.y)
			U = abs(fract(U * 5.0f - iTime * .5f) - .5f) / fwidth(U)
			O = vec4(step(max(U.x, U.y), .3f)) * vec4(.1f, .9f, 1.0f, 1f) * (dot(u, u) * 5.0f)
		}
	}
	shader.render(ivec2(200, 200), 0f)
}

fun test3(out: OutputStream) {
	val shader = klsl(DefaultFrontend(PPMBackend(out))) {
		mainImage {
			var U = (C - .5f * iResolution.xy) / iResolution.y
			val p = U
			val t = iTime * .3f
			U *= mat2(cos(t), sin(t), -sin(t), cos(t))
			U *= sin(iTime) * 3.0f + 4.0f
			val d = length(U) * max(sin(atan(U.y, U.x) * 5.0f + iTime * 2.0f), 0.0f)
			O = vec4(sin(d * 2.0f - t) * .5f + .5f) * exp(-d * .8f) * (1.0f - length(p) * .5f)
		}
	}
	shader.render(ivec2(200, 200), 0f)
}

fun test4(out: OutputStream) {
	val shader = klsl(DefaultFrontend(PPMBackend(out))) {
		mainImage {
			val U = (C - .5f * iResolution.xy) / iResolution.y * 5.0f
			val p = vec3(atan(U.y, U.x) * 3.0f, length(U) * 2.0f, iTime * .5f)
			val h = sin(p.x + cos(p.y) * 2.0f) * .5f + .5f
			O = vec4(h * vec3(1f, .3f, sin(iTime) * .5f + .5f), 1f) * (1.0f - smoothstep(.4f, .41f, abs(mod(p.y, 1.0f) - .5f)))
		}
	}
	shader.render(ivec2(200, 200), 0f)
}

interface TypeDesc<T> {
	val simpleName: String
	fun tryCast(value: Any): T?
}

class vec2(val x: Float, val y: Float) {
	fun self(x: Float, y: Float) = vec2(x, y)

	val yx get() = self(y, x)

	inline fun map(mapper: (Float) -> Float) = self(mapper(x), mapper(y))
	inline fun zipMap(otherX: Float, otherY: Float, mapper: (Float, Float) -> Float) = self(mapper(x, otherX), mapper(y, otherY))
	inline fun zipMap(other: vec2, mapper: (Float, Float) -> Float) = zipMap(other.x, other.y, mapper)

	operator fun plus(v: Float) = map { it + v }
	operator fun minus(v: Float) = map { it - v }
	operator fun times(v: Float) = map { it * v }
	operator fun div(v: Float) = map { it / v }
	operator fun plus(other: vec2) = zipMap(other) { a, b -> a + b }
	operator fun minus(other: vec2) = zipMap(other) { a, b -> a - b }
	operator fun times(other: vec2) = zipMap(other) { a, b -> a * b }
	operator fun div(other: vec2) = zipMap(other) { a, b -> a / b }

	operator fun times(mat: mat2) = self(x * mat.a + y * mat.c, x * mat.b + y * mat.d)

	companion object : TypeDesc<vec2> {
		override val simpleName = "vec2"
		override fun tryCast(value: Any) = value as? vec2
	}
}

class ivec2(val x: Int, val y: Int) {
	fun self(x: Int, y: Int) = ivec2(x, y)

	val yx get() = self(y, x)

	inline fun map(mapper: (Int) -> Int) = self(mapper(x), mapper(y))
	inline fun zipMap(otherX: Int, otherY: Int, mapper: (Int, Int) -> Int) = self(mapper(x, otherX), mapper(y, otherY))
	inline fun zipMap(other: ivec2, mapper: (Int, Int) -> Int) = zipMap(other.x, other.y, mapper)

	operator fun plus(v: Int) = map { it + v }
	operator fun minus(v: Int) = map { it - v }
	operator fun times(v: Int) = map { it * v }
	operator fun div(v: Int) = map { it / v }
	operator fun plus(other: ivec2) = zipMap(other) { a, b -> a + b }
	operator fun minus(other: ivec2) = zipMap(other) { a, b -> a - b }
	operator fun times(other: ivec2) = zipMap(other) { a, b -> a * b }
	operator fun div(other: ivec2) = zipMap(other) { a, b -> a / b }

	companion object : TypeDesc<ivec2> {
		override val simpleName = "ivec2"
		override fun tryCast(value: Any) = value as? ivec2
	}
}

class vec3(val x: Float, val y: Float, val z: Float) {
	fun self(x: Float, y: Float, z: Float) = vec3(x, y, z)

	val xy get() = vec2(x, y)
	val yx get() = vec2(y, x)

	inline fun map(mapper: (Float) -> Float) = self(mapper(x), mapper(y), mapper(z))
	inline fun zipMap(otherX: Float, otherY: Float, otherZ: Float, mapper: (Float, Float) -> Float) = self(mapper(x, otherX), mapper(y, otherY), mapper(z, otherZ))
	inline fun zipMap(other: vec3, mapper: (Float, Float) -> Float) = zipMap(other.x, other.y, other.z, mapper)

	operator fun plus(v: Float) = map { it + v }
	operator fun minus(v: Float) = map { it - v }
	operator fun times(v: Float) = map { it * v }
	operator fun div(v: Float) = map { it / v }
	operator fun plus(other: vec3) = zipMap(other) { a, b -> a + b }
	operator fun minus(other: vec3) = zipMap(other) { a, b -> a - b }
	operator fun times(other: vec3) = zipMap(other) { a, b -> a * b }
	operator fun div(other: vec3) = zipMap(other) { a, b -> a / b }
}

class vec4(val x: Float, val y: Float, val z: Float, val w: Float) {
	constructor(v: Float) : this(v, v, v, v)
	constructor(vec3: vec3, v: Float) : this(vec3.x, vec3.y, vec3.z, v)

	fun self(x: Float, y: Float, z: Float, w: Float) = vec4(x, y, z, w)

	inline fun map(mapper: (Float) -> Float) = self(mapper(x), mapper(y), mapper(z), mapper(w))
	inline fun zipMap(otherX: Float, otherY: Float, otherZ: Float, otherW: Float, mapper: (Float, Float) -> Float) = self(mapper(x, otherX), mapper(y, otherY), mapper(z, otherZ), mapper(w, otherW))
	inline fun zipMap(other: vec4, mapper: (Float, Float) -> Float) = zipMap(other.x, other.y, other.z, other.w, mapper)

	operator fun plus(v: Float) = map { it + v }
	operator fun minus(v: Float) = map { it - v }
	operator fun times(v: Float) = map { it * v }
	operator fun div(v: Float) = map { it / v }
	operator fun plus(other: vec4) = zipMap(other) { a, b -> a + b }
	operator fun minus(other: vec4) = zipMap(other) { a, b -> a - b }
	operator fun times(other: vec4) = zipMap(other) { a, b -> a * b }
	operator fun div(other: vec4) = zipMap(other) { a, b -> a / b }

	companion object : TypeDesc<vec4> {
		override val simpleName = "vec4"
		override fun tryCast(value: Any) = value as? vec4
	}
}

class mat2(val a: Float, val b: Float, val c: Float, val d: Float) {
	operator fun get(i: Int): vec2 = when (i) {
		0 -> vec2(a, b)
		1 -> vec2(c, d)
		else -> throw IndexOutOfBoundsException()
	}

	companion object : TypeDesc<mat2> {
		override val simpleName = "mat2"
		override fun tryCast(value: Any) = value as? mat2
	}
}

class DefaultFrontend(private val backend: Backend) : Frontend {
	override fun <T> uniform(type: TypeDesc<T>, name: String): T {
		val value = _uniforms[name]
			?: throw IllegalArgumentException("Uniform $name is not set")
		val ret = type.tryCast(value)
			?: throw IllegalArgumentException("Uniform $name is not of type ${type.simpleName}")
		return ret
	}

	class MainImageScopeImpl(
		override val C: vec2,
		override val iResolution: vec3,
		override val iTime: Float,
	) : Frontend.MainImageScope {
		override var O: vec4? = null
	}

	override fun mainImage(block: Frontend.MainImageScope.() -> Unit) {
		_mainImageFunction = block
	}

	override fun render(uniforms: Map<String, Any>, dimension: ivec2, time: Float, block: Frontend.() -> Unit) {
		_uniforms = ConcurrentHashMap(uniforms)
		apply(block)
		val pixelCalculator = PixelCalculator { x, y ->
			MainImageScopeImpl(
				vec2(x.toFloat(), y.toFloat()),
				vec3(dimension.x.toFloat(), dimension.y.toFloat(), 1f),
				time,
			).apply(_mainImageFunction!!)
				.O!!
		}
		backend.render(dimension, pixelCalculator)
	}

	// 状态
	internal var _mainImageFunction: (Frontend.MainImageScope.() -> Unit)? = null
	var _uniforms: Map<String, Any> = emptyMap()
}

interface Backend {
	fun render(dimension: ivec2, pixelCalculator: PixelCalculator)
}

fun interface PixelCalculator {
	fun calculate(x: Int, y: Int): vec4
}

interface Frontend {
	fun <T> uniform(type: TypeDesc<T>, name: String): T

	interface MainImageScope {
		val C: vec2
		val iResolution: vec3
		val iTime: Float
		var O: vec4?

		fun abs(v: vec2): vec2 = v.map(::abs)
		fun atan(y: Float, x: Float) = atan(y / x)
		fun fract(x: Float) = x - floor(x)
		fun fract(v: vec2): vec2 = v.map(::fract)
		fun fwidth(v: Float) = abs(dFdx(v)) + abs(dFdy(v))
		fun fwidth(v: vec2) = abs(dFdx(v)) + abs(dFdy(v))
		fun step(edge: Float, x: Float) = if (x < edge) 0f else 1f
		fun dot(v: vec2, w: vec2) = v.x * w.x + v.y * w.y

		fun mod(x: Float, y: Float) = x - y * floor(x / y)
		fun smoothstep(edge0: Float, edge1: Float, x: Float) = when {
			x <= edge0 -> 0f
			x >= edge1 -> 1f

			else -> {
				val t = (x - edge0) / (edge1 - edge0)
				t * t * (3f - 2f * t)
			}
		}

		operator fun Float.times(v: vec2) = v * this
		operator fun Float.times(v: vec3) = v * this
		fun length(v: vec2) = sqrt(v.x * v.x + v.y * v.y)

		// todo
		fun dFdx(v: Float) = 0f
		fun dFdy(v: Float) = 0f
		fun dFdx(v: vec2) = vec2(dFdx(v.x), dFdx(v.y))
		fun dFdy(v: vec2) = vec2(dFdy(v.x), dFdy(v.y))
	}

	fun mainImage(block: MainImageScope.() -> Unit)

	fun render(uniforms: Map<String, Any>, dimension: ivec2, time: Float, block: Frontend.() -> Unit)
}

interface Shader {
	fun setUniform(name: String, value: Any)
	fun render(dimension: ivec2, time: Float)
}

fun klsl(frontend: Frontend, block: Frontend.() -> Unit): Shader {
	return object : Shader {
		private val uniforms = mutableMapOf<String, Any>()

		override fun setUniform(name: String, value: Any) {
			uniforms[name] = value
		}

		override fun render(dimension: ivec2, time: Float) {
			frontend.render(uniforms, dimension, time, block)
		}
	}
}

// single: 11.143092400s
// multi : 1.297833300s
class PPMBackend(private val output: OutputStream) : Backend {
	override fun render(dimension: ivec2, pixelCalculator: PixelCalculator) {
		require(dimension.x > 0 && dimension.y > 0)

		output.write("P6\n${dimension.x} ${dimension.y}\n255\n".encodeToByteArray())

		val pixelCount = dimension.x * dimension.y
		val cpuCount = Runtime.getRuntime().availableProcessors()
		if (pixelCount < cpuCount * cpuCount) {
			for (y in 0..<dimension.y) for (x in 0..<dimension.x) {
				val o = pixelCalculator.calculate(x, y)
				val r = (o.x * 255f).toInt().coerceIn(0, 255)
				val g = (o.y * 255f).toInt().coerceIn(0, 255)
				val b = (o.z * 255f).toInt().coerceIn(0, 255)
				output.write(r)
				output.write(g)
				output.write(b)
			}
			return
		}
		val chunkSize = (pixelCount + cpuCount - 1) / cpuCount
		val lastChunkSize = pixelCount - chunkSize * (cpuCount - 1)
		val chunks = AtomicReferenceArray<ByteArray>(cpuCount)
		IntStream.range(0, cpuCount)
			.parallel()
			.forEach { chunkIndex ->
				val size = if (chunkIndex == cpuCount - 1) lastChunkSize else chunkSize
				val pixelIndex = chunkIndex * chunkSize
				val chunk = ByteArray(size * 3)
				for (i in 0..<size) {
					val x = (pixelIndex + i) % dimension.x
					val y = (pixelIndex + i) / dimension.x
					val o = pixelCalculator.calculate(x, y)
					val r = (o.x * 255f).toInt().coerceIn(0, 255).toByte()
					val g = (o.y * 255f).toInt().coerceIn(0, 255).toByte()
					val b = (o.z * 255f).toInt().coerceIn(0, 255).toByte()
					chunk[i * 3 + 0] = r
					chunk[i * 3 + 1] = g
					chunk[i * 3 + 2] = b
				}
				chunks[chunkIndex] = chunk
			}
		repeat(cpuCount) { output.write(chunks[it]) }
	}
}
