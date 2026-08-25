"""Compute kernels for the Python-facing float64 NumPy subset."""

from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import cos, exp, isnan, log, sin, sqrt, tanh
from std.sys.info import num_physical_cores, simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime PARALLEL_ELEMENTS = 262144
comptime PARALLEL_MATMUL_FLOPS = 1048576


@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int, num_workers: Int):
    for i in range(num_work_items):
        func(i)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


@export("mn_add")
def mn_add(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, a.load[width=W](i) + b.load[width=W](i))
        i += W
    while i < n:
        dst[i] = a[i] + b[i]
        i += 1


@export("mn_subtract")
def mn_subtract(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, a.load[width=W](i) - b.load[width=W](i))
        i += W
    while i < n:
        dst[i] = a[i] - b[i]
        i += 1


@export("mn_multiply")
def mn_multiply(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, a.load[width=W](i) * b.load[width=W](i))
        i += W
    while i < n:
        dst[i] = a[i] * b[i]
        i += 1


@export("mn_divide")
def mn_divide(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, a.load[width=W](i) / b.load[width=W](i))
        i += W
    while i < n:
        dst[i] = a[i] / b[i]
        i += 1


@export("mn_maximum")
def mn_maximum(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        if isnan(a[i]) or isnan(b[i]):
            dst[i] = a[i] + b[i]
        else:
            dst[i] = max(a[i], b[i])


@export("mn_minimum")
def mn_minimum(a_addr: Int, b_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        if isnan(a[i]) or isnan(b[i]):
            dst[i] = a[i] + b[i]
        else:
            dst[i] = min(a[i], b[i])


@export("mn_negative")
def mn_negative(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, -a.load[width=W](i))
        i += W
    while i < n:
        dst[i] = -a[i]
        i += 1


@export("mn_absolute")
def mn_absolute(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, abs(a.load[width=W](i)))
        i += W
    while i < n:
        dst[i] = abs(a[i])
        i += 1


@export("mn_square")
def mn_square(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        var v = a.load[width=W](i)
        dst.store(i, v * v)
        i += W
    while i < n:
        dst[i] = a[i] * a[i]
        i += 1


@export("mn_sqrt")
def mn_sqrt(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, sqrt(a.load[width=W](i)))
        i += W
    while i < n:
        dst[i] = sqrt(a[i])
        i += 1


@export("mn_exp")
def mn_exp(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, exp(a.load[width=W](i)))
        i += W
    while i < n:
        dst[i] = exp(a[i])
        i += 1


@export("mn_log")
def mn_log(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, log(a.load[width=W](i)))
        i += W
    while i < n:
        dst[i] = log(a[i])
        i += 1


@export("mn_sin")
def mn_sin(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, sin(a.load[width=W](i)))
        i += W
    while i < n:
        dst[i] = sin(a[i])
        i += 1


@export("mn_cos")
def mn_cos(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var i = 0
    while i + W <= n:
        dst.store(i, cos(a.load[width=W](i)))
        i += W
    while i < n:
        dst[i] = cos(a[i])
        i += 1


@export("mn_tanh")
def mn_tanh(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    var workers = num_physical_cores() if n >= PARALLEL_ELEMENTS else 1

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        var i = start
        while i + W <= end:
            dst.store(i, tanh(a.load[width=W](i)))
            i += W
        while i < end:
            dst[i] = tanh(a[i])
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mn_clip")
def mn_clip(
    a_addr: Int, dst_addr: Int, n: Int, lo: Float64, hi: Float64
) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[i] = min(max(a[i], lo), hi)


@export("mn_reduce")
def mn_reduce(
    a_addr: Int,
    dst_addr: Int,
    rows: Int,
    cols: Int,
    operation: Int,
    ddof: Int,
) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    for r in range(rows):
        var row = a + r * cols
        if operation == 0 or operation == 4 or operation == 5:
            var acc0 = SIMD[DType.float64, W](0.0)
            var acc1 = SIMD[DType.float64, W](0.0)
            var acc2 = SIMD[DType.float64, W](0.0)
            var acc3 = SIMD[DType.float64, W](0.0)
            var j = 0
            while j + 4 * W <= cols:
                acc0 += row.load[width=W](j)
                acc1 += row.load[width=W](j + W)
                acc2 += row.load[width=W](j + 2 * W)
                acc3 += row.load[width=W](j + 3 * W)
                j += 4 * W
            var vector_total = acc0 + acc1 + acc2 + acc3
            while j + W <= cols:
                vector_total += row.load[width=W](j)
                j += W
            var total = vector_total.reduce_add()
            while j < cols:
                total += row[j]
                j += 1
            if operation == 0:
                dst[r] = total
            elif operation == 4:
                dst[r] = total / Float64(cols)
            else:
                var mean = total / Float64(cols)
                var acc = 0.0
                for j in range(cols):
                    var d = row[j] - mean
                    acc += d * d
                dst[r] = acc / Float64(cols - ddof)
        elif operation == 1:
            var product = 1.0
            for j in range(cols):
                product *= row[j]
            dst[r] = product
        elif operation == 2:
            var value = row[0]
            for j in range(1, cols):
                if isnan(value):
                    break
                if isnan(row[j]) or row[j] < value:
                    value = row[j]
            dst[r] = value
        else:
            var value = row[0]
            for j in range(1, cols):
                if isnan(value):
                    break
                if isnan(row[j]) or row[j] > value:
                    value = row[j]
            dst[r] = value


@export("mn_argreduce")
def mn_argreduce(
    a_addr: Int, dst_addr: Int, rows: Int, cols: Int, operation: Int
) abi("C"):
    var a = fp(a_addr)
    var dst = ip(dst_addr)
    for r in range(rows):
        var row = a + r * cols
        var best = 0
        for j in range(1, cols):
            if isnan(row[best]):
                break
            if isnan(row[j]):
                best = j
                break
            if (operation == 0 and row[j] < row[best]) or (
                operation == 1 and row[j] > row[best]
            ):
                best = j
        dst[r] = Int64(best)


def ordered_before(a: Float64, b: Float64) -> Bool:
    if isnan(a):
        return isnan(b)
    if isnan(b):
        return True
    return a <= b


def strictly_before(a: Float64, b: Float64) -> Bool:
    return a < b


def insertion_sort_values(data: FPtr, lo: Int, hi: Int):
    for i in range(lo + 1, hi + 1):
        var value = data[i]
        var j = i
        while j > lo and strictly_before(value, data[j - 1]):
            data[j] = data[j - 1]
            j -= 1
        data[j] = value


def quicksort_values(data: FPtr, initial_lo: Int, initial_hi: Int):
    var lo = initial_lo
    var hi = initial_hi
    while hi - lo > 24:
        var pivot = data[lo + (hi - lo) // 2]
        var lower = lo
        var i = lo
        var upper = hi
        while i <= upper:
            if strictly_before(data[i], pivot):
                var value = data[lower]
                data[lower] = data[i]
                data[i] = value
                lower += 1
                i += 1
            elif strictly_before(pivot, data[i]):
                var value = data[upper]
                data[upper] = data[i]
                data[i] = value
                upper -= 1
            else:
                i += 1
        if lower - lo < hi - upper:
            if lo < lower - 1:
                quicksort_values(data, lo, lower - 1)
            lo = upper + 1
        else:
            if upper + 1 < hi:
                quicksort_values(data, upper + 1, hi)
            hi = lower - 1
    if lo < hi:
        insertion_sort_values(data, lo, hi)


def merge_numeric(src: FPtr, tmp: FPtr, start: Int, mid: Int, end: Int):
    var i = start
    var j = mid
    var k = start
    while i < mid and j < end:
        if src[i] <= src[j]:
            tmp[k] = src[i]
            i += 1
        else:
            tmp[k] = src[j]
            j += 1
        k += 1
    while i < mid:
        tmp[k] = src[i]
        i += 1
        k += 1
    while j < end:
        tmp[k] = src[j]
        j += 1
        k += 1


@export("mn_sort_inplace")
def mn_sort_inplace(data_addr: Int, n: Int) abi("C"):
    if n > 1:
        quicksort_values(fp(data_addr), 0, n - 1)


@export("mn_merge_numeric")
def mn_merge_numeric(
    src_addr: Int, tmp_addr: Int, start: Int, mid: Int, end: Int
) abi("C"):
    merge_numeric(fp(src_addr), fp(tmp_addr), start, mid, end)


@export("mn_sort")
def mn_sort(
    a_addr: Int, dst_addr: Int, work_addr: Int, rows: Int, cols: Int
) abi("C"):
    var a = fp(a_addr)
    var dst = fp(dst_addr)
    for r in range(rows):
        var dst_row = dst + r * cols
        var input_row = a + r * cols
        var j = 0
        while j + W <= cols:
            dst_row.store(j, input_row.load[width=W](j))
            j += W
        while j < cols:
            dst_row[j] = input_row[j]
            j += 1
        var numeric_end = cols
        j = 0
        while j < numeric_end:
            if isnan(dst_row[j]):
                numeric_end -= 1
                var value = dst_row[numeric_end]
                dst_row[numeric_end] = dst_row[j]
                dst_row[j] = value
            else:
                j += 1
        if numeric_end >= PARALLEL_ELEMENTS:
            var workers = min(num_physical_cores(), numeric_end // 16384)
            var chunk = (numeric_end + workers - 1) // workers
            var chunks = (numeric_end + chunk - 1) // chunk

            @parameter
            def sort_chunk(worker: Int):
                var start = worker * chunk
                var end = min(start + chunk, numeric_end)
                if start < end:
                    quicksort_values(dst_row, start, end - 1)

            parallelize[sort_chunk](chunks, workers)
            var src = dst_row
            var tmp = fp(work_addr) + r * cols
            var width = chunk
            var in_dst = True
            while width < numeric_end:
                var merges = (numeric_end + 2 * width - 1) // (2 * width)
                workers = min(num_physical_cores(), merges)

                @parameter
                def merge_chunk(worker: Int):
                    var first = worker * merges // workers
                    var last = (worker + 1) * merges // workers
                    for merge in range(first, last):
                        var start = merge * 2 * width
                        var mid = min(start + width, numeric_end)
                        var end = min(start + 2 * width, numeric_end)
                        merge_numeric(src, tmp, start, mid, end)

                if workers > 1:
                    parallelize[merge_chunk](workers, workers)
                else:
                    merge_chunk(0)
                var swap = src
                src = tmp
                tmp = swap
                in_dst = not in_dst
                width *= 2
            if not in_dst:
                j = 0
                while j + W <= numeric_end:
                    dst_row.store(j, src.load[width=W](j))
                    j += W
                while j < numeric_end:
                    dst_row[j] = src[j]
                    j += 1
        elif numeric_end > 1:
            quicksort_values(dst_row, 0, numeric_end - 1)


@export("mn_argsort")
def mn_argsort(
    a_addr: Int,
    dst_addr: Int,
    values_addr: Int,
    work_values_addr: Int,
    work_idx_addr: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    var a = fp(a_addr)
    var dst = ip(dst_addr)
    var values = fp(values_addr)
    var work_values = fp(work_values_addr)
    var work_idx = ip(work_idx_addr)
    for r in range(rows):
        var vals = values + r * cols
        var idx = dst + r * cols
        var tmpv = work_values + r * cols
        var tmpi = work_idx + r * cols
        for j in range(cols):
            vals[j] = a[r * cols + j]
            idx[j] = Int64(j)
        var width = 1
        while width < cols:
            var start = 0
            while start < cols:
                var mid = min(start + width, cols)
                var end = min(start + 2 * width, cols)
                var i = start
                var j = mid
                var k = start
                while i < mid and j < end:
                    if ordered_before(vals[i], vals[j]):
                        tmpv[k] = vals[i]
                        tmpi[k] = idx[i]
                        i += 1
                    else:
                        tmpv[k] = vals[j]
                        tmpi[k] = idx[j]
                        j += 1
                    k += 1
                while i < mid:
                    tmpv[k] = vals[i]
                    tmpi[k] = idx[i]
                    i += 1
                    k += 1
                while j < end:
                    tmpv[k] = vals[j]
                    tmpi[k] = idx[j]
                    j += 1
                    k += 1
                start += 2 * width
            for q in range(cols):
                vals[q] = tmpv[q]
                idx[q] = tmpi[q]
            width *= 2


@export("mn_dot")
def mn_dot(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Float64:
    var a = fp(a_addr)
    var b = fp(b_addr)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


@export("mn_matmul")
def mn_matmul(
    a_addr: Int, b_addr: Int, dst_addr: Int, m: Int, k: Int, n: Int
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var workers = (
        min(num_physical_cores(), m)
        if 2 * m * k * n >= PARALLEL_MATMUL_FLOPS else 1
    )

    @parameter
    def process(worker: Int):
        var start = worker * m // workers
        var end = (worker + 1) * m // workers
        for i in range(start, end):
            var row = a + i * k
            var target = dst + i * n
            var j = 0
            while j + 4 <= n:
                var acc0 = SIMD[DType.float64, W](0.0)
                var acc1 = SIMD[DType.float64, W](0.0)
                var acc2 = SIMD[DType.float64, W](0.0)
                var acc3 = SIMD[DType.float64, W](0.0)
                var q = 0
                while q + W <= k:
                    var values = row.load[width=W](q)
                    acc0 += values * (b + j * k).load[width=W](q)
                    acc1 += values * (b + (j + 1) * k).load[width=W](q)
                    acc2 += values * (b + (j + 2) * k).load[width=W](q)
                    acc3 += values * (b + (j + 3) * k).load[width=W](q)
                    q += W
                var total0 = acc0.reduce_add()
                var total1 = acc1.reduce_add()
                var total2 = acc2.reduce_add()
                var total3 = acc3.reduce_add()
                while q < k:
                    var value = row[q]
                    total0 += value * b[j * k + q]
                    total1 += value * b[(j + 1) * k + q]
                    total2 += value * b[(j + 2) * k + q]
                    total3 += value * b[(j + 3) * k + q]
                    q += 1
                target[j] = total0
                target[j + 1] = total1
                target[j + 2] = total2
                target[j + 3] = total3
                j += 4
            while j < n:
                var source = b + j * k
                var acc = SIMD[DType.float64, W](0.0)
                var q = 0
                while q + W <= k:
                    acc += row.load[width=W](q) * source.load[width=W](q)
                    q += W
                var total = acc.reduce_add()
                while q < k:
                    total += row[q] * source[q]
                    q += 1
                target[j] = total
                j += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def matmul_gpu_kernel(
    a: FPtr, b: FPtr, dst: FPtr, m: Int64, k: Int64, n: Int64
):
    var index = Int64(global_idx.x)
    if index >= m * n:
        return
    var i = index // n
    var j = index - i * n
    var total = 0.0
    for q in range(Int(k)):
        total += a[Int(i * k) + q] * b[q * Int(n) + Int(j)]
    dst[Int(index)] = total


@export("mn_matmul_gpu")
def mn_matmul_gpu(
    a_addr: Int, b_addr: Int, dst_addr: Int, m: Int, k: Int, n: Int
) abi("C") -> Int:
    if m <= 0 or k <= 0 or n <= 0:
        return 0
    try:
        with DeviceContext() as ctx:
            var memory = ctx.get_memory_info()
            var elements = m * k + k * n + m * n
            var allocation_bytes = UInt(elements) * UInt(8)
            if memory[0] < UInt(4000 * 1024 * 1024):
                return 0
            if allocation_bytes >= UInt(2 * 1024 * 1024 * 1024):
                return 0
            var device_a = ctx.enqueue_create_buffer[DType.float64](m * k)
            var device_b = ctx.enqueue_create_buffer[DType.float64](k * n)
            var device_dst = ctx.enqueue_create_buffer[DType.float64](m * n)
            ctx.enqueue_copy(device_a, fp(a_addr))
            ctx.enqueue_copy(device_b, fp(b_addr))
            comptime block_size = 256
            var count = m * n
            ctx.enqueue_function[matmul_gpu_kernel](
                device_a,
                device_b,
                device_dst,
                Int64(m),
                Int64(k),
                Int64(n),
                grid_dim=(count + block_size - 1) // block_size,
                block_dim=block_size,
            )
            ctx.enqueue_copy(fp(dst_addr), device_dst)
            ctx.synchronize()
        return 1
    except:
        return 0


@export("mn_norm")
def mn_norm(a_addr: Int, n: Int) abi("C") -> Float64:
    var a = fp(a_addr)
    var scale = 0.0
    var ssq = 1.0
    for i in range(n):
        if a[i] != 0.0:
            var value = abs(a[i])
            if scale < value:
                var ratio = scale / value
                ssq = 1.0 + ssq * ratio * ratio
                scale = value
            else:
                var ratio = value / scale
                ssq += ratio * ratio
    return 0.0 if scale == 0.0 else scale * sqrt(ssq)


@export("mn_cholesky")
def mn_cholesky(a_addr: Int, n: Int) abi("C") -> Int:
    var a = fp(a_addr)
    for i in range(n):
        for j in range(i + 1):
            var acc = a[i * n + j]
            for q in range(j):
                acc -= a[i * n + q] * a[j * n + q]
            if i == j:
                if acc <= 0.0:
                    return 0
                a[i * n + i] = sqrt(acc)
            else:
                a[i * n + j] = acc / a[j * n + j]
        for j in range(i + 1, n):
            a[i * n + j] = 0.0
    return 1


@export("mn_solve")
def mn_solve(a_addr: Int, b_addr: Int, n: Int, nrhs: Int) abi("C") -> Int:
    var a = fp(a_addr)
    var b = fp(b_addr)
    for k in range(n):
        var pivot = k
        for i in range(k + 1, n):
            if abs(a[i * n + k]) > abs(a[pivot * n + k]):
                pivot = i
        if a[pivot * n + k] == 0.0:
            return 0
        if pivot != k:
            for j in range(n):
                var av = a[k * n + j]
                a[k * n + j] = a[pivot * n + j]
                a[pivot * n + j] = av
            for j in range(nrhs):
                var bv = b[k * nrhs + j]
                b[k * nrhs + j] = b[pivot * nrhs + j]
                b[pivot * nrhs + j] = bv
        for i in range(k + 1, n):
            var factor = a[i * n + k] / a[k * n + k]
            a[i * n + k] = factor
            for j in range(k + 1, n):
                a[i * n + j] -= factor * a[k * n + j]
            for j in range(nrhs):
                b[i * nrhs + j] -= factor * b[k * nrhs + j]
    for ri in range(n):
        var i = n - 1 - ri
        for j in range(nrhs):
            var acc = b[i * nrhs + j]
            for q in range(i + 1, n):
                acc -= a[i * n + q] * b[q * nrhs + j]
            b[i * nrhs + j] = acc / a[i * n + i]
    return 1


@export("mn_det")
def mn_det(a_addr: Int, n: Int) abi("C") -> Float64:
    var a = fp(a_addr)
    var result = 1.0
    for k in range(n):
        var pivot = k
        for i in range(k + 1, n):
            if abs(a[i * n + k]) > abs(a[pivot * n + k]):
                pivot = i
        if a[pivot * n + k] == 0.0:
            return 0.0
        if pivot != k:
            result = -result
            for j in range(n):
                var value = a[k * n + j]
                a[k * n + j] = a[pivot * n + j]
                a[pivot * n + j] = value
        var diagonal = a[k * n + k]
        result *= diagonal
        for i in range(k + 1, n):
            var factor = a[i * n + k] / diagonal
            for j in range(k + 1, n):
                a[i * n + j] -= factor * a[k * n + j]
    return result


def next_u64(state: UPtr) -> UInt64:
    var value = state[0] * 6364136223846793005 + 1442695040888963407
    state[0] = value
    return value


def unit_float(state: UPtr) -> Float64:
    return Float64(Int(next_u64(state) >> 11)) / Float64(1 << 53)


def advance_lcg(state: UInt64, delta: Int) -> UInt64:
    var acc_mult = UInt64(1)
    var acc_plus = UInt64(0)
    var cur_mult = UInt64(6364136223846793005)
    var cur_plus = UInt64(1442695040888963407)
    var remaining = delta
    while remaining > 0:
        if remaining & 1:
            acc_mult *= cur_mult
            acc_plus = acc_plus * cur_mult + cur_plus
        cur_plus = (cur_mult + 1) * cur_plus
        cur_mult *= cur_mult
        remaining >>= 1
    return acc_mult * state + acc_plus


def random_normal_range(
    initial_state: UInt64,
    dst: FPtr,
    n: Int,
    start: Int,
    end: Int,
    mean: Float64,
    scale: Float64,
):
    var local_state = advance_lcg(initial_state, 2 * start)
    for pair in range(start, end):
        local_state = (
            local_state * 6364136223846793005 + 1442695040888963407
        )
        var u1 = Float64(Int(local_state >> 11)) / Float64(1 << 53)
        if u1 == 0.0:
            u1 = 1.0 / Float64(1 << 53)
        local_state = (
            local_state * 6364136223846793005 + 1442695040888963407
        )
        var u2 = Float64(Int(local_state >> 11)) / Float64(1 << 53)
        var radius = sqrt(-2.0 * log(u1))
        var angle = 6.283185307179586 * u2
        var i = 2 * pair
        dst[i] = mean + scale * radius * cos(angle)
        if i + 1 < n:
            dst[i + 1] = mean + scale * radius * sin(angle)


@export("mn_random_normal_range")
def mn_random_normal_range(
    state_addr: Int,
    dst_addr: Int,
    n: Int,
    start: Int,
    end: Int,
    mean: Float64,
    scale: Float64,
) abi("C"):
    random_normal_range(
        initial_state=up(state_addr)[0],
        dst=fp(dst_addr),
        n=n,
        start=start,
        end=end,
        mean=mean,
        scale=scale,
    )


@export("mn_random_normal_advance")
def mn_random_normal_advance(state_addr: Int, pairs: Int) abi("C"):
    var state = up(state_addr)
    state[0] = advance_lcg(state[0], 2 * pairs)


@export("mn_random_uniform")
def mn_random_uniform(
    state_addr: Int, dst_addr: Int, n: Int, lo: Float64, hi: Float64
) abi("C"):
    var state = up(state_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[i] = lo + (hi - lo) * unit_float(state)


@export("mn_random_normal")
def mn_random_normal(
    state_addr: Int, dst_addr: Int, n: Int, mean: Float64, scale: Float64
) abi("C"):
    var state = up(state_addr)
    var dst = fp(dst_addr)
    var pairs = (n + 1) // 2
    var workers = num_physical_cores() if n >= PARALLEL_ELEMENTS else 1
    var initial_state = state[0]

    @parameter
    def process(worker: Int):
        var start = worker * pairs // workers
        var end = (worker + 1) * pairs // workers
        random_normal_range(initial_state, dst, n, start, end, mean, scale)

    if workers == 1:
        process(0)
    else:
        parallelize[process](workers, workers)
    state[0] = advance_lcg(initial_state, 2 * pairs)


@export("mn_random_integers")
def mn_random_integers(
    state_addr: Int, dst_addr: Int, n: Int, lo: Int64, hi: Int64
) abi("C"):
    var state = up(state_addr)
    var dst = ip(dst_addr)
    var span = UInt64(hi - lo)
    for i in range(n):
        dst[i] = lo + Int64(next_u64(state) % span)
