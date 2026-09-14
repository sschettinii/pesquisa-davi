"""
Métricas de Avaliação para Redução de Dimensionalidade (Sortedness).

Este módulo implementa em NumPy puro (sem dependência de bibliotecas de alto nível
como scikit-learn ou scipy para os cálculos estatísticos centrais) as métricas propostas no artigo:
"Nonparametric Dimensionality Reduction Quality Assessment based on Sortedness of Unrestricted Neighborhood"
(D. Pereira-Santos et al., EuroVis 2023).

Métricas implementadas:
1. sortedness (Local Neighborhood Sortedness - Equação 1)
2. reciprocal_sortedness (Reciprocal Neighborhood Sortedness - Equação 2)
3. global_pairwise_sortedness (Global Pairwise Sortedness - Equação 3)
4. local_pairwise_sortedness (Local Pairwise Sortedness - Equação 4)

Métricas de referência da literatura (para fins comparativos):
- trustworthiness (Kaski et al., 2003)
- continuity (Kaski et al., 2003)
- kruskal_stress (Fórmula 1 de Kruskal: métrica e não-métrica)
"""

import math
import numpy as np


# ==============================================================================
# 1. FUNÇÕES AUXILIARES DE DISTÂNCIA E ORDENAÇÃO
# ==============================================================================

def sqeuclidean_distance_matrix(X):
    """
    Calcula a matriz de distâncias euclidianas ao quadrado entre todos os pontos de X.
    
    Fórmula:
        ||x_i - x_j||^2 = <x_i, x_i> + <x_j, x_j> - 2 * <x_i, x_j>
    
    Parâmetros
    ----------
    X : np.ndarray de shape (n, d)
        Matriz de dados (n observações, d dimensões).
        
    Retorna
    -------
    D2 : np.ndarray de shape (n, n)
        Matriz simétrica contendo as distâncias euclidianas ao quadrado.
    """
    # Soma dos quadrados das coordenadas de cada linha: shape (n, 1)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    
    # Expansão: ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j^T
    D2 = sq_norms + sq_norms.T - 2.0 * np.dot(X, X.T)
    
    # Devido a pequenas imprecisões numéricas de ponto flutuante, valores podem ficar ligeiramente negativos
    np.maximum(D2, 0.0, out=D2)
    
    # A distância de um ponto para ele mesmo é rigorosamente zero
    np.fill_diagonal(D2, 0.0)
    return D2


def euclidean_distance_matrix(X):
    """
    Calcula a matriz de distâncias euclidianas entre todos os pontos de X.
    
    Parâmetros
    ----------
    X : np.ndarray de shape (n, d)
    
    Retorna
    -------
    D : np.ndarray de shape (n, n)
    """
    return np.sqrt(sqeuclidean_distance_matrix(X))


def rank_data_1d(a):
    """
    Atribui ranks (postos) aos elementos de um vetor 1D, tratando empates pela média (método 'average').
    Ranks retornados são base 0 (o menor elemento recebe 0, ou a média dos primeiros ranks em caso de empate).
    
    Parâmetros
    ----------
    a : np.ndarray 1D
    
    Retorna
    -------
    ranks : np.ndarray 1D com os ranks em float64
    """
    a = np.asarray(a)
    n = len(a)
    order = np.argsort(a)
    sorted_a = a[order]
    
    ranks = np.empty(n, dtype=np.float64)
    
    i = 0
    while i < n:
        j = i
        # Identifica bloco de elementos idênticos (empates)
        while j < n - 1 and sorted_a[j + 1] == sorted_a[j]:
            j += 1
        
        # Média dos ranks base 0 atribuídos a esse bloco: [i, i+1, ..., j]
        avg_rank = (i + j) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
        
    return ranks


def get_lexicographical_rank(x, y):
    """
    Calcula o ranking lexicográfico decrescente por (x, y).
    
    Regra:
    - Elementos com maior valor de x recebem maior prioridade (rank 0 é a maior prioridade).
    - Empates em x são desempatados usando o maior valor de y.
    - Isso corresponde exatamente à regra adotada por Sebastiano Vigna (2015) e SciPy.
    
    Retorna
    -------
    rank : np.ndarray de inteiros (0 a n-1)
    """
    # np.lexsort ordena crescentemente pela última chave passada (chave primária é a última).
    # Como queremos ordem decrescente de x, desempatada por decrescente de y:
    # chave secundária = -y, chave primária = -x
    order = np.lexsort((-y, -x))
    
    # Inverte a permutação: descobre a posição (rank) de cada elemento original
    rank = np.empty_like(order, dtype=np.intp)
    rank[order] = np.arange(len(order))
    return rank


# ==============================================================================
# 2. ÍNDICES DE CORRELAÇÃO DE KENDALL TAU (PADRÃO E PONDERADO)
# ==============================================================================

def kendall_tau(x, y):
    """
    Calcula o coeficiente de correlação de postos de Kendall tau-b entre dois vetores x e y.
    Também calcula o p-valor assintótico para testar a hipótese nula de ausência de correlação (tau = 0).
    
    Fórmula de Kendall tau-b:
        tau = (P - Q) / sqrt((P + Q + Tx) * (P + Q + Ty))
    onde:
        - P: número de pares concordantes: (x_i - x_j) * (y_i - y_j) > 0
        - Q: número de pares discordantes: (x_i - x_j) * (y_i - y_j) < 0
        - Tx: empates apenas em x
        - Ty: empates apenas em y
        
    Parâmetros
    ----------
    x, y : arrays unidimensionais de mesmo tamanho
    
    Retorna
    -------
    tau : float
        Valor da correlação no intervalo [-1, 1].
    p_value : float
        P-valor bilateral sob a hipótese nula H0: ausência de correlação.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)
    
    if n < 2:
        return np.nan, np.nan
        
    # Sinais das diferenças entre todos os pares (i, j)
    # diff_x[i, j] = sign(x_i - x_j)
    diff_x = np.sign(x[:, None] - x[None, :])
    diff_y = np.sign(y[:, None] - y[None, :])
    
    # Consideramos apenas o triângulo superior (pares i < j sem repetição)
    triu_idx = np.triu_indices(n, k=1)
    sgn_x = diff_x[triu_idx]
    sgn_y = diff_y[triu_idx]
    
    # Classificação dos pares
    prod = sgn_x * sgn_y
    P = np.sum(prod > 0)    # Concordantes
    Q = np.sum(prod < 0)    # Discordantes
    
    # Empates
    Tx = np.sum((sgn_x == 0) & (sgn_y != 0))  # Empate em x mas não em y
    Ty = np.sum((sgn_x != 0) & (sgn_y == 0))  # Empate em y mas não em x
    
    n_pairs = len(sgn_x)
    denom = np.sqrt(float(P + Q + Tx) * float(P + Q + Ty))
    
    if denom == 0.0:
        return 0.0, 1.0
        
    tau = (P - Q) / denom
    
    # Cálculo do p-valor assintótico sob a distribuição nula
    # Variância de S = (P - Q) para n observações
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    
    # Ajuste para empates na variância (se houver empates idênticos)
    # Para dados contínuos, a aproximação padrão é altamente precisa
    z = (P - Q) / math.sqrt(var_s)
    
    # p-valor bilateral a partir da função erro complementar (distribuição normal padrão)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    
    return float(tau), float(p_value)


def vigna_weighted_tau(x, y, rank, additive=True):
    """
    Calcula o Kendall tau ponderado conforme proposto por Sebastiano Vigna (2015):
    "A weighted correlation index for rankings with ties".
    
    Estrutura Matemática:
    --------------------
    Dado um vetor de rankings de importância `rank` (onde 0 é o elemento mais importante),
    a função de ponderação hiperbólica associa a cada elemento o peso:
        w(r) = 1 / (r + 1)
        
    O peso atribuído à comparação entre o par de elementos (i, j) é:
        W(i, j) = w(rank[i]) + w(rank[j])    (versão aditiva hiperbólica)
        
    A covariância ponderada é dada por:
        <x, y>_{rank, w} = sum_{i < j} sgn(x_i - x_j) * sgn(y_i - y_j) * W(i, j)
        
    As normas ponderadas são:
        ||x||_{rank, w} = sqrt( sum_{i < j} [sgn(x_i - x_j)]^2 * W(i, j) )
        ||y||_{rank, w} = sqrt( sum_{i < j} [sgn(y_i - y_j)]^2 * W(i, j) )
        
    O coeficiente de correlação é a "similaridade de cosseno" ponderada:
        tau_w = <x, y> / ( ||x|| * ||y|| )
        
    Propriedades:
    - tau_w está rigorosamente no intervalo [-1, 1] devido à desigualdade de Cauchy-Schwarz.
    - Se dois elementos empatam em x (x_i == x_j), sgn(x_i - x_j) = 0, penalizando a norma.
    - Trocas entre vizinhos próximos (baixo valor de rank) têm peso MUITO superior
      a trocas entre pontos distantes.
    
    Parâmetros
    ----------
    x, y : arrays 1D de escores ou distâncias invertidas
    rank : array 1D com o rank de importância de cada elemento (0 = mais prioritário)
    additive : bool (se True, W(i, j) = w_i + w_j; se False, W(i, j) = w_i * w_j)
    
    Retorna
    -------
    tau_w : float no intervalo [-1, 1]
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    rank = np.asarray(rank).ravel()
    n = len(x)
    
    if n < 2:
        return 1.0
        
    # Ponderação hiperbólica conforme Equação 1 do artigo: w(i) = 1 / (i + 1)
    w = 1.0 / (rank + 1.0)
    
    # Sinais das diferenças entre pares
    diff_x = np.sign(x[:, None] - x[None, :])
    diff_y = np.sign(y[:, None] - y[None, :])
    
    # Matriz de pesos dos pares W(i, j)
    if additive:
        W = w[:, None] + w[None, :]
    else:
        W = w[:, None] * w[None, :]
        
    # Apenas pares únicos i < j
    triu_idx = np.triu_indices(n, k=1)
    sgn_x = diff_x[triu_idx]
    sgn_y = diff_y[triu_idx]
    w_pairs = W[triu_idx]
    
    # Covariância e variâncias ponderadas
    cov = np.sum(sgn_x * sgn_y * w_pairs)
    var_x = np.sum((sgn_x ** 2) * w_pairs)
    var_y = np.sum((sgn_y ** 2) * w_pairs)
    
    denom = np.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
        
    return float(cov / denom)


# ==============================================================================
# 3. MÉTRICAS PROPOSTAS NO ARTIGO
# ==============================================================================

def sortedness(X, X_proj, i=None, symmetric=True, weighted=True):
    """
    1. Local Neighborhood Sortedness - lambda_tau_w(x) (Equação 1 do artigo).
    
    Avalia o grau de concordância na ordenação dos vizinhos em relação a cada ponto x.
    
    Como funciona:
    -------------
    Para cada ponto x_k:
    1. Calcula as distâncias de x_k para todos os outros pontos u != x_k no espaço original (D_orig)
       e no espaço projetado (D_proj).
    2. Define os escores como distâncias negativas: escores maiores indicam pontos MAIS PRÓXIMOS.
    3. Constrói o ranking de importância a partir das distâncias (vizinho mais próximo recebe rank 0).
       Em caso de empate, o desempate lexicográfico é aplicado.
    4. Se `symmetric=True`, calcula tanto a ênfase na extrusão (ponderada pelas distâncias originais)
       quanto a ênfase na intrusão (ponderada pelas distâncias projetadas) e toma a média aritmética:
           lambda(x) = 0.5 * (tau_w(orig, proj | rank_orig) + tau_w(proj, orig | rank_proj))
    5. Se `weighted=False`, calcula a variante não ponderada lambda_tau_1, análoga ao stress não métrico.
    
    Parâmetros
    ----------
    X : np.ndarray de shape (N, D)
        Conjunto de dados original em alta dimensão.
    X_proj : np.ndarray de shape (N, d)
        Conjunto de dados projetado em baixa dimensão (ex: 2D).
    i : int ou None
        Se fornecido (índice do ponto), calcula apenas para o ponto de interesse x_i.
        Se None, calcula para todos os N pontos.
    symmetric : bool
        Se True, faz a média entre extrusão e intrusão. Se False, pondera apenas pelo espaço original.
    weighted : bool
        Se True, usa a ponderação hiperbólica w(r) = 1/(r+1) de Vigna.
        Se False, usa pesos constantes w(r) = 1 (Kendall tau tradicional).
        
    Retorna
    -------
    lambda_tau : np.ndarray de shape (N,) ou float (se i for fornecido)
        Valores de sortedness no intervalo [-1, 1].
        1.0: preservação perfeita da vizinhança irrestrita.
        0.0: ordenação aleatória (nenhuma informação preservada).
       -1.0: inversão perfeita da vizinhança.
    """
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    N = len(X)
    
    D_orig_sq = sqeuclidean_distance_matrix(X)
    D_proj_sq = sqeuclidean_distance_matrix(X_proj)
    
    indices = [i] if i is not None else range(N)
    resultados = []
    
    for idx in indices:
        # Máscara booleana para excluir o próprio ponto da sua lista de vizinhos
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        
        # Distâncias do ponto idx para todos os outros pontos
        d_orig = D_orig_sq[idx, mask]
        d_proj = D_proj_sq[idx, mask]
        
        if not weighted:
            # Variante não ponderada lambda_tau_1: Kendall tau padrão entre as distâncias
            tau, _ = kendall_tau(d_orig, d_proj)
            resultados.append(tau)
            continue
            
        # Para o Kendall tau ponderado: escores maiores = maior proximidade (daí o sinal negativo)
        scores_orig = -d_orig
        scores_proj = -d_proj
        
        # Ênfase na Extrusão (ranking pelos dados originais)
        rank_orig = get_lexicographical_rank(scores_orig, scores_proj)
        tau_extrusion = vigna_weighted_tau(scores_orig, scores_proj, rank_orig)
        
        if symmetric:
            # Ênfase na Intrusão (ranking pelos dados projetados)
            rank_proj = get_lexicographical_rank(scores_proj, scores_orig)
            tau_intrusion = vigna_weighted_tau(scores_proj, scores_orig, rank_proj)
            tau_final = 0.5 * (tau_extrusion + tau_intrusion)
        else:
            tau_final = tau_extrusion
            
        resultados.append(tau_final)
        
    if i is not None:
        return float(resultados[0])
    return np.array(resultados, dtype=np.float64)


def reciprocal_sortedness(X, X_proj, i=None, symmetric=True):
    """
    2. Reciprocal Sortedness - lambda_tau_w^rec(x) (Equação 2 do artigo).
    
    Considera a relação de vizinhança na perspectiva recíproca (conceito de hubness).
    
    Como funciona:
    -------------
    Em vez de avaliar "quem são os vizinhos de x", avalia:
    "Qual a posição (rank) de x na lista de vizinhos de cada um dos OUTROS pontos?"
    
    Para cada ponto u != x:
    - O ponto u ordena todos os seus vizinhos por distância.
    - Descobre-se a posição r_u(x) que x ocupa na lista de u.
    - Isso gera um vetor de ranks recíprocos de x em relação a todos os demais pontos.
    - Compara-se o vetor de ranks recíprocos no espaço original versus no projetado via tau_w.
    
    Parâmetros
    ----------
    X, X_proj : matrizes de dados original e projetada
    i : int ou None
    symmetric : bool
    
    Retorna
    -------
    lambda_rec : np.ndarray de shape (N,) ou float
    """
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    N = len(X)
    
    D_orig_sq = sqeuclidean_distance_matrix(X)
    D_proj_sq = sqeuclidean_distance_matrix(X_proj)
    
    # R[i, j] deve conter o rank do ponto i na vizinhança do ponto j (coluna j)
    R_orig = np.zeros((N, N), dtype=np.float64)
    R_proj = np.zeros((N, N), dtype=np.float64)
    
    for j in range(N):
        R_orig[:, j] = rank_data_1d(D_orig_sq[:, j])
        R_proj[:, j] = rank_data_1d(D_proj_sq[:, j])
        
    indices = [i] if i is not None else range(N)
    resultados = []
    
    for idx in indices:
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        
        # Posições de idx nas listas de todos os outros pontos
        # Ranks menores significam maior proximidade do ponto idx perante os outros
        # Negativamos para que "mais próximo" corresponda a "maior escore"
        scores_orig = -R_orig[idx, mask]
        scores_proj = -R_proj[idx, mask]
        
        rank_orig = get_lexicographical_rank(scores_orig, scores_proj)
        tau_extrusion = vigna_weighted_tau(scores_orig, scores_proj, rank_orig)
        
        if symmetric:
            rank_proj = get_lexicographical_rank(scores_proj, scores_orig)
            tau_intrusion = vigna_weighted_tau(scores_proj, scores_orig, rank_proj)
            tau_final = 0.5 * (tau_extrusion + tau_intrusion)
        else:
            tau_final = tau_extrusion
            
        resultados.append(tau_final)
        
    if i is not None:
        return float(resultados[0])
    return np.array(resultados, dtype=np.float64)


def global_pairwise_sortedness(X, X_proj):
    """
    3. Global Pairwise Sortedness - Lambda_tau_1(X, X_hat) (Equação 3 do artigo).
    
    Avalia a ordenação global de todas as distâncias entre pares de pontos.
    
    Como funciona:
    -------------
    1. Calcula todas as combinações de distâncias entre pares (u, v) com u < v (total M = N*(N-1)/2 pares).
    2. Compara o ranking de distâncias RX no espaço original contra o ranking R_hat_X no projetado.
    3. Utiliza o Kendall tau não ponderado (w = 1).
    4. Fornece um p-valor para a hipótese nula de ordenação aleatória (Lambda_tau_1 = 0).
    
    Sensibilidade:
    - É sensível a distorções sutis na estrutura global:
      Ex: se d(a, b) > d(b, c) no original, mas d_hat(a, b) < d_hat(b, c) na projeção,
      a métrica é penalizada, mesmo que a ordem local de vizinhos mais próximos tenha sido mantida.
      
    Retorna
    -------
    tau : float no intervalo [-1, 1]
    p_value : float
    """
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    N = len(X)
    
    triu_idx = np.triu_indices(N, k=1)
    
    D_orig_sq = sqeuclidean_distance_matrix(X)
    D_proj_sq = sqeuclidean_distance_matrix(X_proj)
    
    # Vetores de distâncias de todos os pares (u, v) com u < v
    d_orig_pairs = D_orig_sq[triu_idx]
    d_proj_pairs = D_proj_sq[triu_idx]
    
    return kendall_tau(d_orig_pairs, d_proj_pairs)


def local_pairwise_sortedness(X, X_proj, i=None, symmetric=True):
    """
    4. Local Pairwise Sortedness - Lambda_tau_w(x) (Equação 4 do artigo).
    
    Generalização ponderada do pairwise sortedness para avaliação ponto a ponto.
    
    Como funciona:
    -------------
    Para um ponto de referência x_k:
    1. Consideramos todos os M = N*(N-1)/2 pares (u, v) do conjunto de dados.
    2. A importância de cada par (u, v) para x_k é proporcional à proximidade média do par até x_k:
           dist_media((u, v), x_k) = (d(u, x_k) + d(v, x_k)) / 2
    3. Ordenamos todos os pares pela sua proximidade média até x_k (pares mais próximos de x_k
       recebem os primeiros postos, com maior peso hiperbólico).
    4. Calculamos o Kendall tau ponderado entre as distâncias de todos os pares no original e no projetado.
    5. Se `symmetric=True`, fazemos a média entre o ranking de pares baseado nas distâncias originais
       e o baseado nas distâncias projetadas.
       
    Retorna
    -------
    lambda_pw : np.ndarray de shape (N,) ou float
    """
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    N = len(X)
    
    D_orig_sq = sqeuclidean_distance_matrix(X)
    D_proj_sq = sqeuclidean_distance_matrix(X_proj)
    
    triu_idx = np.triu_indices(N, k=1)
    u_idx, v_idx = triu_idx
    
    # Distâncias de todos os pares (invertidas para que maior escore = par mais próximo)
    scores_orig = -D_orig_sq[triu_idx]
    scores_proj = -D_proj_sq[triu_idx]
    
    # Matrizes de proximidade do par (u, v) até cada ponto de referência k:
    # M_orig[p, k] = d(u, k) + d(v, k)
    M_orig = D_orig_sq[u_idx, :] + D_orig_sq[v_idx, :]
    M_proj = D_proj_sq[u_idx, :] + D_proj_sq[v_idx, :]
    
    indices = [i] if i is not None else range(N)
    resultados = []
    
    for k in indices:
        # Ranking dos pares pela proximidade a x_k
        # Menor valor de M_orig[:, k] = par mais próximo do ponto k = rank 0
        order_orig = np.argsort(M_orig[:, k])
        rank_orig = np.empty_like(order_orig, dtype=np.float64)
        rank_orig[order_orig] = np.arange(len(order_orig), dtype=np.float64)
        
        tau_extrusion = vigna_weighted_tau(scores_orig, scores_proj, rank_orig)
        
        if symmetric:
            order_proj = np.argsort(M_proj[:, k])
            rank_proj = np.empty_like(order_proj, dtype=np.float64)
            rank_proj[order_proj] = np.arange(len(order_proj), dtype=np.float64)
            
            tau_intrusion = vigna_weighted_tau(scores_orig, scores_proj, rank_proj)
            tau_final = 0.5 * (tau_extrusion + tau_intrusion)
        else:
            tau_final = tau_extrusion
            
        resultados.append(tau_final)
        
    if i is not None:
        return float(resultados[0])
    return np.array(resultados, dtype=np.float64)


# ==============================================================================
# 4. MÉTRICAS CLÁSSICAS DE REFERÊNCIA (COMPARAÇÃO COM A LITERATURA)
# ==============================================================================

def trustworthiness(X, X_proj, k=5):
    """
    Trustworthiness (Confiabilidade) - Kaski et al. (2003).
    
    Mede a precisão das vizinhanças de baixa dimensão em relação a intrusões
    (pontos que parecem próximos na projeção 2D, mas eram distantes no espaço original).
    
    Fórmula:
        T(k) = 1 - (2 / (N * k * (2N - 3k - 1))) * sum_{i=1}^N sum_{j in U_i^{(k)}} (r(i, j) - k)
    onde:
        - U_i^{(k)}: conjunto de pontos que estão entre os k vizinhos mais próximos de i
          no espaço projetado, mas NÃO estavam entre os k vizinhos no original.
        - r(i, j): posto (base 0) do ponto j na vizinhança de i no espaço original.
        
    Retorna
    -------
    T : float no intervalo [0, 1]
    """
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    N = len(X)
    
    if k >= N - 1:
        return 1.0
        
    D_orig = euclidean_distance_matrix(X)
    D_proj = euclidean_distance_matrix(X_proj)
    
    # Preenche a diagonal com infinito para não considerar o próprio ponto
    np.fill_diagonal(D_orig, np.inf)
    np.fill_diagonal(D_proj, np.inf)
    
    # Ranks dos pontos no espaço original: base 0
    # r_orig[i, j] é a posição do ponto j em relação ao ponto i
    order_orig = np.argsort(D_orig, axis=1)
    ranks_orig = np.empty_like(order_orig)
    row_idx = np.arange(N)[:, None]
    ranks_orig[row_idx, order_orig] = np.arange(N)
    
    # Vizinhos top-k no espaço projetado
    order_proj = np.argsort(D_proj, axis=1)
    knn_proj = order_proj[:, :k]
    
    # Penalização de intrusão
    penalidade = 0.0
    for i in range(N):
        # Pontos que estão no top-k projetado
        for j in knn_proj[i]:
            rank_ij = ranks_orig[i, j]
            # Se no original o rank era >= k, significa que era um falso vizinho
            if rank_ij >= k:
                penalidade += (rank_ij - k)
                
    constante = 2.0 / (N * k * (2 * N - 3 * k - 1))
    return float(1.0 - constante * penalidade)


def continuity(X, X_proj, k=5):
    """
    Continuity (Continuidade) - Kaski et al. (2003).
    
    Métrica dual do Trustworthiness: mede a perda de vizinhos originais (extrusões).
    Penaliza pontos que eram vizinhos próximos no original mas foram afastados na projeção.
    
    Retorna
    -------
    C : float no intervalo [0, 1]
    """
    # Continuidade é formalmente idêntica ao Trustworthiness invertendo os papéis de X e X_proj
    return trustworthiness(X_proj, X, k=k)


def kruskal_stress(X, X_proj, metric=True):
    """
    Kruskal Stress (Fórmula 1 de Kruskal).
    
    Parâmetros
    ----------
    X, X_proj : matrizes original e projetada
    metric : bool
        Se True, usa as distâncias euclidianas normalizadas (Stress Métrico sigma_1).
        Se False, usa os postos das distâncias normalizados (Stress Não-Métrico sigma^*).
        
    Retorna
    -------
    stress : float no intervalo [0, 1]
        Menor valor indica melhor qualidade.
    """
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    N = len(X)
    
    triu_idx = np.triu_indices(N, k=1)
    
    if metric:
        D_orig = euclidean_distance_matrix(X)[triu_idx]
        D_proj = euclidean_distance_matrix(X_proj)[triu_idx]
    else:
        D_orig = rank_data_1d(euclidean_distance_matrix(X)[triu_idx])
        D_proj = rank_data_1d(euclidean_distance_matrix(X_proj)[triu_idx])
        
    # Normalização das distâncias pelo máximo
    d_max_orig = np.max(D_orig)
    d_max_proj = np.max(D_proj)
    
    if d_max_orig > 0:
        D_orig = D_orig / d_max_orig
    if d_max_proj > 0:
        D_proj = D_proj / d_max_proj
        
    residuos = np.sum((D_orig - D_proj) ** 2)
    escala = np.sum(D_proj ** 2)
    
    if escala == 0:
        return 0.0
        
    return float(np.sqrt(residuos / escala))
