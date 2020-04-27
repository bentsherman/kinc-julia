using CSV
using DataFrames
using CUDAdrv
using CUDAnative
using CuArrays



CLUSMETHOD_NONE = 1
CLUSMETHOD_GMM = 2

CRITERION_AIC = 1
CRITERION_BIC = 2
CRITERION_ICL = 3

CORRMETHOD_PEARSON = 1
CORRMETHOD_SPEARMAN = 2



function fetch_pair(x, y, min_expression, max_expression, labels)
    # label the pairwise samples
    N = 0
    
    for i in 1:length(x)
        # label samples with missing values
        if isnan(x[i]) || isnan(y[i])
            labels[i] = -9

        # label samples which are below the minimum expression threshold
        elseif x[i] < min_expression || y[i] < min_expression
            labels[i] = -6

        # label samples which are above the maximum expression threshold
        elseif x[i] > max_expression || y[i] > max_expression
            labels[i] = -6

        # label any remaining samples as cluster 0
        else
            N += 1
            labels[i] = 0
        end
    end

    # return number of clean samples
    return N
end



function next_power_2(n)
    pow2 = 2
    while pow2 < n
        pow2 *= 2
    end
    return pow2
end



function swap(array, i, j)
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp
end



function bitonic_sort(array)
    n = length(array)
    bsize = div(n, 2)

    ob = 2
    while ob <= n
        ib = ob
        while ib >= 2
            t = div(ib, 2)
            for i in 0:(bsize - 1)
                dir = -(div(i, div(ob, 2)) & 0x1)
                a = div(i, t) * ib + (i % t) + 1
                b = a + t
                if (dir == 0 && (array[a] > array[b])) || (dir != 0 && (array[a] < array[b]))
                    swap(array, a, b)
                end
            end
            ib = div(ib, 2)
        end
        ob *= 2
    end
end



function bitonic_sort(array, extra)
    n = length(array)
    bsize = div(n, 2)

    ob = 2
    while ob <= n
        ib = ob
        while ib >= 2
            t = div(ib, 2)
            for i in 0:(bsize - 1)
                dir = -(div(i, div(ob, 2)) & 0x1)
                a = div(i, t) * ib + (i % t) + 1
                b = a + t
                if (dir == 0 && (array[a] > array[b])) || (dir != 0 && (array[a] < array[b]))
                    swap(array, a, b)
                    swap(extra, a, b)
                end
            end
            ib = div(ib, 2)
        end
        ob *= 2
    end
end



function mark_outliers(x, y, labels, k, marker, x_sorted, y_sorted)
    # extract samples in cluster k
    n = 0
    
    for i in 1:length(x)
        if labels[i] == k
            n += 1
            x_sorted[n] = x[i]
            y_sorted[n] = y[i]
        end
    end

    for i in (n + 1):length(x_sorted)
        x_sorted[i] = Inf
        y_sorted[i] = Inf
    end

    # make sure cluster is not empty
    if n == 0
        return 0
    end

    # sort arrays
    bitonic_sort(x_sorted)
    bitonic_sort(y_sorted)

    # compute quartiles and thresholds for each axis
    Q1_x = x_sorted[1 + div(n * 1, 4)]
    Q3_x = x_sorted[1 + div(n * 3, 4)]
    T_x_min = Q1_x - 1.5 * (Q3_x - Q1_x)
    T_x_max = Q3_x + 1.5 * (Q3_x - Q1_x)

    Q1_y = y_sorted[1 + div(n * 1, 4)]
    Q3_y = y_sorted[1 + div(n * 3, 4)]
    T_y_min = Q1_y - 1.5 * (Q3_y - Q1_y)
    T_y_max = Q3_y + 1.5 * (Q3_y - Q1_y)

    # mark outliers
    n = 0

    for i in 1:length(labels)
        if labels[i] == k
            outlier_x = (x[i] < T_x_min || T_x_max < x[i])
            outlier_y = (y[i] < T_y_min || T_y_max < y[i])

            if outlier_x || outlier_y
                labels[i] = marker
            elseif labels[i] >= 0
                n += 1
            end
        end
    end

    # return number of remaining samples
    return n
end



mutable struct Vector2
    x::Float32
    y::Float32
end



Vector2(v::Vector2) = Vector2(v.x, v.y)



function vector_add(a::Vector2, b::Vector2)
    a.x += b.x
    a.y += b.y
end



function vector_add(a::Vector2, c::Float32, b::Vector2)
    a.x += c * b.x
    a.y += c * b.y
end



function vector_subtract(a::Vector2, b::Vector2)
    a.x -= b.x
    a.y -= b.y
end



function vector_scale(a::Vector2, c::Float32)
    a.x *= c
    a.y *= c
end



function vector_dot(a::Vector2, b::Vector2)
    return a.x * b.x + a.y * b.y
end



function vector_diff_norm(a::Vector2, b::Vector2)
    return sqrt((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2)
end



mutable struct Matrix2x2
    m00::Float32
    m01::Float32
    m10::Float32
    m11::Float32
end



function matrix_scale(A::Matrix2x2, c::Float32)
    A.m00 *= c
    A.m01 *= c
    A.m10 *= c
    A.m11 *= c
end



function matrix_determinant(A::Matrix2x2)
    return A.m00 * A.m11 - A.m01 * A.m10
end



function matrix_inverse(A::Matrix2x2, B::Matrix2x2, det::Float32)
    B.m00 = +A.m11 / det
    B.m01 = -A.m01 / det
    B.m10 = -A.m10 / det
    B.m11 = +A.m00 / det
end



function matrix_product(A::Matrix2x2, x::Vector2, b::Vector2)
    b.x = A.m00 * x.x + A.m01 * x.y
    b.y = A.m10 * x.x + A.m11 * x.y
end



function matrix_add_outer_product(A::Matrix2x2, c::Float32, x::Vector2)
    A.m00 += c * x.x * x.x
    A.m01 += c * x.x * x.y
    A.m10 += c * x.y * x.x
    A.m11 += c * x.y * x.y
end



mutable struct RandomState
    state::UInt64
end



function myrand!(rs::RandomState)
    rs.state = rs.state * 1103515245 + 12345
    return div(rs.state, 65536) % 32768
end



struct GMM_cpu
    data       ::Array{Vector2}
    labels     ::Array{Int8}
    pi         ::Array{Float32}
    mu         ::Array{Vector2}
    sigma      ::Array{Matrix2x2}
    sigmaInv   ::Array{Matrix2x2}
    normalizer ::Array{Float32}
    MP         ::Array{Vector2}
    counts     ::Array{Int32}
    logpi      ::Array{Float32}
    gamma      ::Array{Float32, 2}
    logL       ::Array{Float32}
    entropy    ::Array{Float32}
end



struct GMM_gpu
    data       ::CuArray{Vector2}
    labels     ::CuArray{Int8}
    pi         ::CuArray{Float32}
    mu         ::CuArray{Vector2}
    sigma      ::CuArray{Matrix2x2}
    sigmaInv   ::CuArray{Matrix2x2}
    normalizer ::CuArray{Float32}
    MP         ::CuArray{Vector2}
    counts     ::CuArray{Int32}
    logpi      ::CuArray{Float32}
    gamma      ::CuArray{Float32, 2}
    logL       ::CuArray{Float32}
    entropy    ::CuArray{Float32}
end



function gmm_initialize_components(gmm, X, N, K)
    # initialize random state
    rs = RandomState(1)

    # initialize each mixture component
    for k in 1:K
        # initialize mixture weight to uniform distribution
        gmm.pi[k] = 1.0f0 / K

        # initialize mean to a random sample from X
        i = myrand!(rs) % N + 1

        gmm.mu[k] = Vector2(X[i])

        # initialize covariance to identity matrix
        gmm.sigma[k] = Matrix2x2(1, 0, 0, 1)
        gmm.sigmaInv[k] = Matrix2x2(1, 0, 0, 1)
    end
end



function gmm_prepare_components(gmm, K)
    D = 2

    for k in 1:K
        # compute determinant of covariance matrix
        det = matrix_determinant(gmm.sigma[k])

        # return failure if matrix inverse failed
        if det <= 0.0 || isnan(det)
            return false
        end

        # compute precision matrix (inverse of covariance matrix)
        matrix_inverse(gmm.sigma[k], gmm.sigmaInv[k], det)

        # compute normalizer term for multivariate normal distribution
        gmm.normalizer[k] = -0.5f0 * (D * log(2.0f0 * pi) + log(det))
    end

    return true
end



function gmm_initialize_means(gmm, X, N, K)
    max_iterations = 20
    tolerance = 1f-3

    # initialize workspace
    MP = gmm.MP
    counts = gmm.counts

    for t in 1:max_iterations
        # compute mean and sample count for each component
        for k in 1:K
            MP[k] = Vector2(0, 0)
            counts[k] = 0
        end

        for i in 1:N
            # determine the component mean which is nearest to x_i
            min_dist = Inf
            min_k = 0
            for k in 1:K
                dist = vector_diff_norm(X[i], gmm.mu[k])
                if min_dist > dist
                    min_dist = dist
                    min_k = k
                end
            end

            # update mean and sample count
            vector_add(MP[min_k], X[i])
            counts[min_k] += 1
        end

        # scale each mean by its sample count
        for k in 1:K
            if counts[k] > 0
                vector_scale(MP[k], 1.0f0 / counts[k])
            end
        end

        # compute the total change of all means
        diff = 0.0f0

        for k in 1:K
            diff += vector_diff_norm(MP[k], gmm.mu[k])
        end

        diff /= K
    
        # update component means
        for k in 1:K
            gmm.mu[k] = Vector2(MP[k])
        end
        
        # stop if converged
        if diff < tolerance
            break
        end
    end
end



function gmm_compute_estep(gmm, X, N, K)
    # compute logpi
    for k in 1:K
        gmm.logpi[k] = log(gmm.pi[k])
    end

    # compute the log-probability for each component and each point in X
    logProb = gmm.gamma
    
    for i in 1:N
        for k in 1:K
            # compute xm = (x - mu)
            xm = Vector2(X[i])
            vector_subtract(xm, gmm.mu[k])

            # compute Sxm = Sigma^-1 xm
            Sxm = Vector2(0, 0)
            matrix_product(gmm.sigmaInv[k], xm, Sxm)

            # compute xmSxm = xm^T Sigma^-1 xm
            xmSxm = vector_dot(xm, Sxm)
            
            # compute log(P) = normalizer - 0.5 * xm^T * Sigma^-1 * xm
            logProb[i, k] = gmm.normalizer[k] - 0.5f0 * xmSxm
        end
    end

    # compute gamma and log-likelihood
    logL = 0.0f0
    
    for i in 1:N
        # compute a = argmax(logpi_k + logProb_ik, k)
        maxArg = -Inf
        for k in 1:K
            arg = gmm.logpi[k] + logProb[i, k]
            if maxArg < arg
                maxArg = arg
            end
        end

        # compute logpx
        sum_ = 0.0f0
        for k in 1:K
            sum_ += exp(gmm.logpi[k] + logProb[i, k] - maxArg)
        end

        logpx = maxArg + log(sum_)

        # compute gamma_ik
        for k in 1:K
            gmm.gamma[i, k] += gmm.logpi[k] - logpx
            gmm.gamma[i, k] = exp(gmm.gamma[i, k])
        end

        # update log-likelihood
        logL += logpx
    end

    # return log-likelihood
    return logL
end



function gmm_compute_mstep(gmm, X, N, K)
    for k in 1:K
        # compute n_k = sum(gamma_ik)
        n_k = 0.0f0
        
        for i in 1:N
            n_k += gmm.gamma[i, k]
        end

        # update mixture weight
        gmm.pi[k] = n_k / N

        # update mean
        gmm.mu[k] = Vector2(0, 0)

        for i in 1:N
            vector_add(gmm.mu[k], gmm.gamma[i, k], X[i])
        end

        vector_scale(gmm.mu[k], 1.0f0 / n_k)

        # update covariance matrix
        gmm.sigma[k] = Matrix2x2(0, 0, 0, 0)

        for i in 1:N
            # compute xm = (x_i - mu_k)
            xm = Vector2(X[i])
            vector_subtract(xm, gmm.mu[k])

            # compute Sigma_ki = gamma_ik * (x_i - mu_k) (x_i - mu_k)^T
            matrix_add_outer_product(gmm.sigma[k], gmm.gamma[i, k], xm)
        end

        matrix_scale(gmm.sigma[k], 1.0f0 / n_k)
    end
end



function gmm_compute_labels(gamma, N, K, labels)
    for i in 1:N
        # determine the value k for which gamma_ik is highest
        max_k = -1
        max_gamma = -Inf

        for k in 1:K
            if max_gamma < gamma[i, k]
                max_k = k - 1
                max_gamma = gamma[i, k]
            end
        end

        # assign x_i to cluster k
        labels[i] = max_k
    end
end



function gmm_compute_entropy(gamma, N, labels)
    E = 0.0f0
    
    for i in 1:N
        k = labels[i]
        E -= log(gamma[i, k + 1])
    end
    
    return E
end



function gmm_fit(gmm, X, N, K, labels)
    # initialize mixture components
    gmm_initialize_components(gmm, X, N, K)

    # initialize means with k-means
    gmm_initialize_means(gmm, X, N, K)

    # run EM algorithm
    max_iterations = 100
    tolerance = 1f-8
    prevLogL = -Inf
    currLogL = -Inf

    for t in 1:max_iterations
        # pre-compute precision matrix and normalizer term for each mixture component
        success = gmm_prepare_components(gmm, K)

        # return failure if matrix inverse failed
        if !success
            return false
        end

        # perform E step
        prevLogL = currLogL
        currLogL = gmm_compute_estep(gmm, X, N, K)

        # check for convergence
        if abs(currLogL - prevLogL) < tolerance
            break
        end

        # perform M step
        gmm_compute_mstep(gmm, X, N, K)
    end

    # save outputs
    gmm.logL[1] = currLogL
    gmm_compute_labels(gmm.gamma, N, K, labels)
    gmm.entropy[1] = gmm_compute_entropy(gmm.gamma, N, labels)

    return true
end



function compute_aic(K, D, logL)
    p = K * (1 + D + D * D)
    
    return 2 * p - 2 * logL
end



function compute_bic(K, D, logL, N)
    p = K * (1 + D + D * D)
    
    return log(N) * p - 2 * logL
end



function compute_icl(K, D, logL, N, E)
    p = K * (1 + D + D * D)

    return log(N) * p - 2 * logL + 2 * E
end



function gmm_compute(
    gmm,
    x, y,
    n_samples,
    labels,
    min_samples,
    min_clusters,
    max_clusters,
    criterion)

    # perform clustering only if there are enough samples
    bestK = 0

    if n_samples >= min_samples
        # extract clean samples from data array
        j = 1
        for i in 1:length(x)
            if labels[i] >= 0
                gmm.data[j] = Vector2(x[i], y[i])
                j += 1
            end
        end

        # determine the number of clusters
        bestValue = Inf

        for K in min_clusters:max_clusters
            # run the clustering model
            success = gmm_fit(gmm, gmm.data, n_samples, K, gmm.labels)

            if !success
                continue
            end

            # compute the criterion value of the model
            value = Inf

            if criterion == CRITERION_AIC
                value = compute_aic(K, 2, gmm.logL[1])
            elseif criterion == CRITERION_BIC
                value = compute_bic(K, 2, gmm.logL[1], n_samples)
            elseif criterion == CRITERION_ICL
                value = compute_icl(K, 2, gmm.logL[1], n_samples, gmm.entropy[1])
            end

            # save the model with the lowest criterion value
            if value < bestValue
                bestK = K
                bestValue = value

                # save labels for clean samples
                j = 1
                for i in 1:length(x)
                    if labels[i] >= 0
                        labels[i] = gmm.labels[j]
                        j += 1
                    end
                end
            end
        end
    end

    return bestK
end



function pearson(x, y, labels, k, min_samples)
    n = 0
    sumx = 0
    sumy = 0
    sumx2 = 0
    sumy2 = 0
    sumxy = 0

    for i in 1:length(x)
        if labels[i] == k
            x_i = x[i]
            y_i = y[i]

            sumx += x_i
            sumy += y_i
            sumx2 += x_i * x_i
            sumy2 += y_i * y_i
            sumxy += x_i * y_i

            n += 1
        end
    end

    if n >= min_samples
        return (n*sumxy - sumx*sumy) / sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy))
    end

    return NaN
end



function compute_rank(array)
    n = length(array)
    i = 1

    while i < n
        a_i = array[i]

        if a_i == array[i + 1]
            j = i + 2
            rank = 0.0

            # we have detected a tie, find number of equal elements
            while j < n && a_i == array[j]
                j += 1
            end

            # compute rank
            for k in i:(j - 1)
                rank += k
            end

            # divide by number of ties
            rank /= (j - i)

            for k in i:(j - 1)
                array[k] = rank
            end

            i = j
        else
            # no tie - set rank to natural ordered position
            array[i] = i
            i += 1
        end
    end

    if i == n
        array[n] = n
    end
end



function spearman(x, y, labels, k, min_samples, x_rank, y_rank)
    # extract samples in pairwise cluster
    n = 0

    for i in 1:length(x)
        if labels[i] == k
            n += 1
            x_rank[n] = x[i]
            y_rank[n] = y[i]
        end
    end

    # get power of 2 size
    for i in (n + 1):length(x_rank)
        x_rank[i] = Inf
        y_rank[i] = Inf
    end

    # compute correlation only if there are enough samples
    if n >= min_samples
        # compute rank of x
        bitonic_sort(x_rank, y_rank)
        compute_rank(x_rank)

        # compute rank of y
        bitonic_sort(y_rank, x_rank)
        compute_rank(y_rank)

        # compute correlation of rank arrays
        sumx = 0
        sumy = 0
        sumx2 = 0
        sumy2 = 0
        sumxy = 0

        for i in 1:n
            x_i = x_rank[i]
            y_i = y_rank[i]

            sumx += x_i
            sumy += y_i
            sumx2 += x_i * x_i
            sumy2 += y_i * y_i
            sumxy += x_i * y_i
        end

        return (n*sumxy - sumx*sumy) / sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy))
    end

    return NaN
end



function similarity_kernel(
    x, y,
    clusmethod,
    corrmethod,
    preout,
    postout,
    min_expression,
    max_expression,
    min_samples,
    min_clusters,
    max_clusters,
    criterion,
    x_sorted,
    y_sorted,
    gmm,
    labels,
    correlations)

    # fetch pairwise data
    n_samples = fetch_pair(
        x, y,
        min_expression,
        max_expression,
        labels)

    # remove pre-clustering outliers
    if preout
        n_samples = mark_outliers(
            x, y,
            labels,
            0,
            -7,
            x_sorted,
            y_sorted)
    end

    # perform clustering
    K = 1

    if clusmethod == CLUSMETHOD_GMM
        K = gmm_compute(
            gmm,
            x, y,
            n_samples,
            labels,
            min_samples,
            min_clusters,
            max_clusters,
            criterion)
    end

    # remove post-clustering outliers
    if K > 1 && postout
        for k in 1:K
            n_samples = mark_outliers(
                x, y,
                labels,
                k - 1,
                -8,
                x_sorted,
                y_sorted)
        end
    end

    # perform correlation
    if corrmethod == CORRMETHOD_PEARSON
        for k in 1:K
            correlations[k] = pearson(
                x, y,
                labels,
                k - 1,
                min_samples)
        end

    elseif corrmethod == CORRMETHOD_SPEARMAN
        for k in 1:K
            correlations[k] = spearman(
                x, y,
                labels,
                k - 1,
                min_samples,
                x_sorted,
                y_sorted)
        end
    end

    # return number of clusters
    return K
end



function write_pair(
    i, j,
    K,
    labels,
    correlations,
    mincorr,
    maxcorr,
    outfile)

    # determine number of valid correlations
    valid = [(!isnan(r) && mincorr <= abs(r) && abs(r) <= maxcorr) for r in correlations]
    n_clusters = sum(valid)
    cluster_idx = 1

    # write each correlation to output file
    for k in 1:K
        corr = correlations[k]

        # make sure correlation meets thresholds
        if valid[k]
            # compute sample mask
            y_k = copy(labels)

            # y_k[(y_k >= 0) & (y_k != k)] = 0
            # y_k[y_k == k] = 1
            # y_k[y_k < 0] *= -1
            for i in 1:length(y_k)
                if y_k[i] >= 0 && y_k[i] != k - 1
                    y_k[i] = 0
                end
                if y_k[i] == k - 1
                    y_k[i] = 1
                end
                if y_k[i] < 0
                    y_k[i] *= -1
                end
            end

            sample_mask = join(y_k, "")

            # compute summary statistics
            n_samples = sum([y == 1 for y in y_k])

            # write correlation to output file
            join(outfile, [i - 1, j - 1, cluster_idx, n_clusters, n_samples, corr, sample_mask], "\t")
            write(outfile, "\n")

            # increment cluster index
            cluster_idx += 1
        end
    end
end



function similarity_cpu(
    emx,
    clusmethod,
    corrmethod,
    preout,
    postout,
    minexpr,
    maxexpr,
    minsamp,
    minclus,
    maxclus,
    criterion,
    mincorr,
    maxcorr,
    outfile)

    # initialize workspace
    N = size(emx, 2)
    N_pow2 = next_power_2(N)
    K = maxclus

    x_sorted = Array{Float32}(undef, N_pow2)
    y_sorted = Array{Float32}(undef, N_pow2)

    gmm = GMM_cpu(
        #= data =#       Array{Vector2}(undef, N),
        #= labels =#     Array{Int8}(undef, N),
        #= pi =#         Array{Float32}(undef, K),
        #= mu =#         Array{Vector2}(undef, K),
        #= sigma =#      Array{Matrix2x2}(undef, K),
        #= sigmaInv =#   Array{Matrix2x2}(undef, K),
        #= normalizer =# Array{Float32}(undef, K),
        #= MP =#         Array{Vector2}(undef, K),
        #= counts =#     Array{Int32}(undef, K),
        #= logpi =#      Array{Float32}(undef, K),
        #= gamma =#      Array{Float32}(undef, N, K),
        #= logL =#       Array{Float32}(undef, 1),
        #= entropy =#    Array{Float32}(undef, 1)
    )

    labels = Array{Int8}(undef, N)
    correlations = Array{Float32}(undef, K)

    # process each gene pair
    for i in 1:size(emx, 1)
        # println(i)

        for j in 1:(i - 1)
            # extract pairwise data
            x = emx[i, :]
            y = emx[j, :]

            # compute pairwise similarity
            K = similarity_kernel(
                x, y,
                clusmethod,
                corrmethod,
                preout,
                postout,
                minexpr,
                maxexpr,
                minsamp,
                minclus,
                maxclus,
                criterion,
                x_sorted,
                y_sorted,
                gmm,
                labels,
                correlations)

            # save pairwise results
            write_pair(
                i, j,
                K,
                labels,
                correlations,
                mincorr,
                maxcorr,
                outfile)
        end
    end
end



struct PairwiseIndex
    x::Int64
    y::Int64
end



function pairwise_increment(index)
    x = index.x
    y = index.y + 1
    if x == y
        x += 1
        y = 1
    end
    return PairwiseIndex(x, y)
end



function similarity_gpu_helper(
    n_pairs,
    in_emx,
    in_index,
    clusmethod,
    corrmethod,
    preout,
    postout,
    minexpr,
    maxexpr,
    minsamp,
    minclus,
    maxclus,
    criterion,
    work_x,
    work_y,
    work_gmm_data,
    work_gmm_labels,
    work_gmm_pi,
    work_gmm_mu,
    work_gmm_sigma,
    work_gmm_sigmaInv,
    work_gmm_normalizer,
    work_gmm_MP,
    work_gmm_counts,
    work_gmm_logpi,
    work_gmm_gamma,
    work_gmm_logL,
    work_gmm_entropy,
    out_K,
    out_labels,
    out_correlations)

    # get global index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i > n_pairs
        return
    end

    # initialize workspace variables
    index = in_index[i]
    x = in_emx[index.x, :]
    y = in_emx[index.y, :]
    x_sorted = work_x[index.x, :]
    y_sorted = work_y[index.y, :]

    gmm = GMM_gpu(
        #= data =#       work_gmm_data[i, :],
        #= labels =#     work_gmm_labels[i, :],
        #= pi =#         work_gmm_pi[i, :],
        #= mu =#         work_gmm_mu[i, :],
        #= sigma =#      work_gmm_sigma[i, :],
        #= sigmaInv =#   work_gmm_sigmaInv[i, :],
        #= normalizer =# work_gmm_normalizer[i, :],
        #= MP =#         work_gmm_MP[i, :],
        #= counts =#     work_gmm_counts[i, :],
        #= logpi =#      work_gmm_logpi[i, :],
        #= gamma =#      work_gmm_gamma[i, :, :],
        #= logL =#       work_gmm_logL[i, :],
        #= entropy =#    work_gmm_entropy[i, :]
    )

    labels = out_labels[i, :]
    correlations = out_correlations[i, :]

    # save number of clusters
    out_K[i] = similarity_kernel(
        x, y,
        clusmethod,
        corrmethod,
        preout,
        postout,
        minexpr,
        maxexpr,
        minsamp,
        minclus,
        maxclus,
        criterion,
        x_sorted,
        y_sorted,
        gmm,
        labels,
        correlations)

    return
end



function similarity_gpu(
    emx,
    clusmethod,
    corrmethod,
    preout,
    postout,
    minexpr,
    maxexpr,
    minsamp,
    minclus,
    maxclus,
    criterion,
    mincorr,
    maxcorr,
    gsize,
    lsize,
    outfile)

    # allocate device buffers
    W = gsize
    N = size(emx, 2)
    N_pow2 = next_power_2(N)
    K = maxclus

    in_emx               = CuArray(emx)
    in_index_cpu         = Array{PairwiseIndex}(undef, W)
    in_index_gpu         = CuArray(in_index_cpu)
    work_x               = CuArray{Float32}(undef, W, N_pow2)
    work_y               = CuArray{Float32}(undef, W, N_pow2)
    work_gmm_data        = CuArray{Vector2}(undef, W, N)
    work_gmm_labels      = CuArray{Int8}(undef, W, N)
    work_gmm_pi          = CuArray{Float32}(undef, W, K)
    work_gmm_mu          = CuArray{Vector2}(undef, W, K)
    work_gmm_sigma       = CuArray{Matrix2x2}(undef, W, K)
    work_gmm_sigmaInv    = CuArray{Matrix2x2}(undef, W, K)
    work_gmm_normalizer  = CuArray{Float32}(undef, W, K)
    work_gmm_MP          = CuArray{Vector2}(undef, W, K)
    work_gmm_counts      = CuArray{Int32}(undef, W, K)
    work_gmm_logpi       = CuArray{Float32}(undef, W, K)
    work_gmm_gamma       = CuArray{Float32}(undef, W, N, K)
    work_gmm_logL        = CuArray{Float32}(undef, W, 1)
    work_gmm_entropy     = CuArray{Float32}(undef, W, 1)
    out_K_cpu            = Array{Int8}(undef, W)
    out_K_gpu            = CuArray(out_K_cpu)
    out_labels_cpu       = Array{Int8}(undef, W, N)
    out_labels_gpu       = CuArray(out_labels_cpu)
    out_correlations_cpu = Array{Float32}(undef, W, K)
    out_correlations_gpu = CuArray(out_correlations_cpu)

    # iterate through global work blocks
    n_genes = size(emx, 1)
    n_total_pairs = div(n_genes * (n_genes - 1), 2)

    base_index = PairwiseIndex(2, 1)

    for i in 1 : gsize : n_total_pairs
        # println(i, " ", n_total_pairs)

        # determine number of pairs
        n_pairs = min(gsize, n_total_pairs - i + 1)

        # initialize index array
        index = base_index

        for j in 1:n_pairs
            in_index_cpu[j] = index
            index = pairwise_increment(index)
        end

        # copy index array to device
        copyto!(in_index_gpu, in_index_cpu)

        # execute similarity kernel
        @cuda blocks=div(gsize, lsize) threads=lsize similarity_gpu_helper(
            n_pairs,
            in_emx,
            in_index_gpu,
            clusmethod,
            corrmethod,
            preout,
            postout,
            minexpr,
            maxexpr,
            minsamp,
            minclus,
            maxclus,
            criterion,
            work_x,
            work_y,
            work_gmm_data,
            work_gmm_labels,
            work_gmm_pi,
            work_gmm_mu,
            work_gmm_sigma,
            work_gmm_sigmaInv,
            work_gmm_normalizer,
            work_gmm_MP,
            work_gmm_counts,
            work_gmm_logpi,
            work_gmm_gamma,
            work_gmm_logL,
            work_gmm_entropy,
            out_K_gpu,
            out_labels_gpu,
            out_correlations_gpu
        )
        CUDAdrv.synchronize()

        # copy results from device
        copyto!(out_K_cpu, out_K_gpu)
        copyto!(out_labels_cpu, out_labels_gpu)
        copyto!(out_correlations_cpu, out_correlations_gpu)

        # save correlation matrix to output file
        index = base_index

        for j in 1:n_pairs
            # extract pairwise results
            K = out_K_cpu[j]
            labels = out_labels_cpu[j]
            correlations = out_correlations_cpu[j, 1:K]

            # save pairwise results
            write_pair(
                index.x,
                index.y,
                K,
                labels,
                correlations,
                mincorr,
                maxcorr,
                outfile)

            # increment pairwise index
            index = pairwise_increment(index)
        end

        # update local pairwise index
        base_index = index
    end
end



function main()
    if length(ARGS) != 3
        println("usage: ./kinc-julia.jl <infile> <outfile> <gpu>")
        return
    end

    # define input parameters
    args_input = ARGS[1]
    args_output = ARGS[2]
    args_gpu = parse(Bool, ARGS[3])
    args_clusmethod = CLUSMETHOD_GMM
    args_corrmethod = CORRMETHOD_SPEARMAN
    args_preout = true
    args_postout = true
    args_minexpr = 0.0
    args_maxexpr = 20.0
    args_minsamp = 30
    args_minclus = 1
    args_maxclus = 5
    args_criterion = CRITERION_ICL
    args_mincorr = 0.5
    args_maxcorr = 1.0
    args_gsize = 4096
    args_lsize = 32

    # load input data
    csv = CSV.read(args_input, delim='\t', missingstring="NA")
    csv = coalesce.(csv, NaN)
    disallowmissing!(csv)

    emx = Matrix{Float32}(csv[:, 2:size(csv, 2)])

    # initialize output file
    outfile = open(args_output, "w")

    # run similarity
    if args_gpu
        similarity_gpu(
            emx,
            args_clusmethod,
            args_corrmethod,
            args_preout,
            args_postout,
            args_minexpr,
            args_maxexpr,
            args_minsamp,
            args_minclus,
            args_maxclus,
            args_criterion,
            args_mincorr,
            args_maxcorr,
            args_gsize,
            args_lsize,
            outfile
        )

    else
        similarity_cpu(
            emx,
            args_clusmethod,
            args_corrmethod,
            args_preout,
            args_postout,
            args_minexpr,
            args_maxexpr,
            args_minsamp,
            args_minclus,
            args_maxclus,
            args_criterion,
            args_mincorr,
            args_maxcorr,
            outfile
        )
    end
end



main()