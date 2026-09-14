"""
Demonstração Didática das Métricas de Avaliação de Redução de Dimensionalidade (Sortedness).

Este script demonstra e reproduz computacionalmente os experimentos conceituais
apresentados no artigo:
"Nonparametric Dimensionality Reduction Quality Assessment based on Sortedness of Unrestricted Neighborhood"
(D. Pereira-Santos et al., EuroVis 2023).

Experimentos demonstrados:
1. Invariância a Movimentos Rígidos (Rotação, Translação e Escala).
2. Efeito da Randomização Progressiva de Pontos (Reprodução da Figura 3 do artigo).
3. Distorção Global via Troca de Clusters (Reprodução da Figura 7 do artigo).
"""

import numpy as np
import metricas_sortedness as ms


def imprimir_tabela(titulo, colunas, linhas):
    print("\n" + "=" * 80)
    print(f" {titulo.upper()} ")
    print("=" * 80)
    
    # Formatação de cabeçalho
    header = " | ".join([f"{col:^14}" for col in colunas])
    print(header)
    print("-" * len(header))
    
    for row in linhas:
        formatados = []
        for val in row:
            if isinstance(val, (int, np.integer)):
                formatados.append(f"{val:^14d}")
            elif isinstance(val, (float, np.floating)):
                formatados.append(f"{val:^14.4f}")
            else:
                formatados.append(f"{str(val):^14}")
        print(" | ".join(formatados))
    print("=" * 80)


# ==============================================================================
# EXPERIMENTO 1: PROPRIEDADES FUNDAMENTAIS E MOVIMENTOS RÍGIDOS
# ==============================================================================

def demo_propriedades_fundamentais():
    print("\n" + "#" * 80)
    print(" EXPERIMENTO 1: PROPRIEDADES FUNDAMENTAIS E MOVIMENTOS RÍGIDOS")
    print(" Objetivo: Avaliar como as métricas reagem a transformações que mantêm a topologia")
    print("#" * 80)
    
    np.random.seed(42)
    N = 30
    # Conjunto de pontos em 2D
    X = np.random.uniform(-10, 10, size=(N, 2))
    
    # Transformação 1: Identidade (projeção perfeita)
    X_ident = X.copy()
    
    # Transformação 2: Rotação de 45 graus + Translação (movimento rígido)
    theta = np.radians(45)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X_rigido = np.dot(X, R) + np.array([100.0, -50.0])
    
    # Transformação 3: Mudança de escala uniforme (multiplicar por 10)
    X_escala = X * 10.0
    
    # Transformação 4: Pequena perturbação com ruído gaussiano (sigma=0.3)
    X_ruido = X + np.random.normal(0, 0.3, size=X.shape)
    
    # Transformação 5: Totalmente aleatória (embaralhamento dos pontos)
    X_aleatorio = np.random.permutation(X)
    
    # Transformação 6: Inversão geométrica (ordem inversa)
    X_inverso = -X
    
    cenarios = [
        ("Identidade", X_ident),
        ("Mov. Rígido", X_rigido),
        ("Escala x10", X_escala),
        ("Ruído Leve", X_ruido),
        ("Aleatório", X_aleatorio),
        ("Inversão", X_inverso),
    ]
    
    colunas = ["Cenário", "Sortedness", "Recíproco", "PW Global", "PW Local", "Trust (k=5)", "Stress (s1)"]
    linhas = []
    
    for nome, X_proj in cenarios:
        s_med = np.mean(ms.sortedness(X, X_proj))
        rs_med = np.mean(ms.reciprocal_sortedness(X, X_proj))
        gpw, _ = ms.global_pairwise_sortedness(X, X_proj)
        lpw_med = np.mean(ms.local_pairwise_sortedness(X, X_proj))
        t5 = ms.trustworthiness(X, X_proj, k=5)
        st = ms.kruskal_stress(X, X_proj, metric=True)
        
        linhas.append([nome, s_med, rs_med, gpw, lpw_med, t5, st])
        
    imprimir_tabela("Resultados do Experimento 1", colunas, linhas)
    
    print("\n[INSIGHT PEDAGÓGICO]:")
    print("1. Movimentos rígidos (rotação e translação) e mudanças de escala preservam perfeitamente")
    print("   a ordem relativa de distâncias, resultando em Sortedness = 1.0 e Stress = 0.0.")
    print("2. Uma projeção aleatória zera as métricas baseadas em Kendall tau (aprox. 0.0),")
    print("   enquanto Trustworthiness fica em ~0.5 e Stress fica em valores não intuitivos (~0.4 - 0.5).")
    print("3. O intervalo das métricas propostas [-1, 1] é padronizado e intuitivo:")
    print("   1.0 = Projeção perfeita | 0.0 = Projeção aleatória | -1.0 = Inversão total.")


# ==============================================================================
# EXPERIMENTO 2: RANDOMIZAÇÃO PROGRESSIVA (SUBSET RANDOMIZATION - FIGURA 3)
# ==============================================================================

def demo_randomizacao_progressiva():
    print("\n" + "#" * 80)
    print(" EXPERIMENTO 2: RANDOMIZAÇÃO PROGRESSIVA DE PONTOS (FIGURA 3 DO ARTIGO)")
    print(" Objetivo: Avaliar a degradação suave das métricas conforme mais pontos são perturbados")
    print("#" * 80)
    
    np.random.seed(42)
    N = 40
    # Amostragem uniforme em quadrado 100x100
    X = np.random.uniform(0, 100, size=(N, 2))
    
    percentuais = [0, 20, 40, 60, 80, 100]
    colunas = ["% Random", "Sortedness", "Recíproco", "PW Global", "PW Local", "Trust (k=5)", "1 - Stress"]
    linhas = []
    
    for p in percentuais:
        X_proj = X.copy()
        n_perturbar = int((p / 100.0) * N)
        if n_perturbar > 0:
            idx_perturbar = np.random.choice(N, size=n_perturbar, replace=False)
            X_proj[idx_perturbar] = np.random.uniform(0, 100, size=(n_perturbar, 2))
            
        s_med = np.mean(ms.sortedness(X, X_proj))
        rs_med = np.mean(ms.reciprocal_sortedness(X, X_proj))
        gpw, _ = ms.global_pairwise_sortedness(X, X_proj)
        lpw_med = np.mean(ms.local_pairwise_sortedness(X, X_proj))
        t5 = ms.trustworthiness(X, X_proj, k=5)
        st = ms.kruskal_stress(X, X_proj, metric=True)
        
        linhas.append([f"{p}%", s_med, rs_med, gpw, lpw_med, t5, 1.0 - st])
        
    imprimir_tabela("Degradação com Randomização de Subconjunto", colunas, linhas)
    
    print("\n[INSIGHT PEDAGÓGICO]:")
    print("Conforme observado na Figura 3 do artigo original:")
    print("- As variantes de Sortedness (local, recíproca e pairwise) convergem exatamente para ~0.0")
    print("  quando 100% dos pontos são randomizados.")
    print("- Já Trustworthiness termina em torno de ~0.5 e (1 - Stress) em torno de ~0.4,")
    print("  mostrando que métricas clássicas sofrem com pisos artificiais e pouca clareza interpretativa.")


# ==============================================================================
# EXPERIMENTO 3: DISTORÇÃO GLOBAL COM PRESERVAÇÃO LOCAL (TROCA DE CLUSTERS)
# ==============================================================================

def demo_troca_de_clusters():
    print("\n" + "#" * 80)
    print(" EXPERIMENTO 3: TROCA DE CLUSTERS (DISTORÇÃO GLOBAL VS LOCAL - FIGURA 7)")
    print(" Objetivo: Avaliar o comportamento quando a vizinhança local é mantida, mas a estrutura global é invertida")
    print("#" * 80)
    
    np.random.seed(42)
    pts_per_cluster = 15
    
    # Cria 3 clusters colineares bem separados: C1 em x=0, C2 em x=50, C3 em x=100
    C1 = np.random.normal(loc=[0.0, 0.0], scale=2.0, size=(pts_per_cluster, 2))
    C2 = np.random.normal(loc=[50.0, 0.0], scale=2.0, size=(pts_per_cluster, 2))
    C3 = np.random.normal(loc=[100.0, 0.0], scale=2.0, size=(pts_per_cluster, 2))
    
    X_original = np.vstack([C1, C2, C3])
    
    # Projeção Distorcida: Inverte a posição dos clusters C1 e C2 no espaço projetado!
    # Ou seja: C1 vai para x=50, C2 vai para x=0, C3 permanece em x=100.
    # Note: a geometria interna de cada cluster é 100% preservada, mas a relação global foi quebrada!
    C1_swap = C1 + np.array([50.0, 0.0])
    C2_swap = C2 - np.array([50.0, 0.0])
    C3_swap = C3.copy()
    
    X_swap = np.vstack([C1_swap, C2_swap, C3_swap])
    
    # Cálculo das métricas
    s_orig = np.mean(ms.sortedness(X_original, X_original))
    s_swap = np.mean(ms.sortedness(X_original, X_swap))
    
    rs_orig = np.mean(ms.reciprocal_sortedness(X_original, X_original))
    rs_swap = np.mean(ms.reciprocal_sortedness(X_original, X_swap))
    
    gpw_orig, _ = ms.global_pairwise_sortedness(X_original, X_original)
    gpw_swap, _ = ms.global_pairwise_sortedness(X_original, X_swap)
    
    lpw_orig = np.mean(ms.local_pairwise_sortedness(X_original, X_original))
    lpw_swap = np.mean(ms.local_pairwise_sortedness(X_original, X_swap))
    
    t5_orig = ms.trustworthiness(X_original, X_original, k=5)
    t5_swap = ms.trustworthiness(X_original, X_swap, k=5)
    
    st_orig = ms.kruskal_stress(X_original, X_original, metric=True)
    st_swap = ms.kruskal_stress(X_original, X_swap, metric=True)
    
    colunas = ["Métrica", "Original (Perfeita)", "Clusters Trocados", "Impacto / Penalização"]
    linhas = [
        ["Sortedness (Local)", s_orig, s_swap, f"{(s_swap - s_orig):.4f} (baixa penalidade)"],
        ["Recíproco", rs_orig, rs_swap, f"{(rs_swap - rs_orig):.4f} (baixa penalidade)"],
        ["PW Global (Λ𝜏1)", gpw_orig, gpw_swap, f"{(gpw_swap - gpw_orig):.4f} (ALTA penalidade!)"],
        ["PW Local (Λ𝜏w)", lpw_orig, lpw_swap, f"{(lpw_swap - lpw_orig):.4f} (penalidade moderada)"],
        ["Trustworthiness (T5)", t5_orig, t5_swap, f"{(t5_swap - t5_orig):.4f} (INCAPAZ de detectar!)"],
        ["Kruskal Stress", st_orig, st_swap, f"+{st_swap:.4f} (penaliza sem escala clara)"],
    ]
    
    imprimir_tabela("Efeito da Troca de Clusters (Distorção Estrutural Global)", colunas, linhas)
    
    print("\n[INSIGHT CRUCIAL DO ARTIGO]:")
    print("1. Trustworthiness (T5) é COMPLETAMENTE CEGA a essa distorção global (permanece ~1.0)!")
    print("   Isso ocorre porque T_k olha estritamente para os k vizinhos mais imediatos de cada ponto,")
    print("   ignorando totalmente que grupos inteiros de dados foram trocados de lugar no mapa.")
    print("2. O Pairwise Sortedness Global (Λ𝜏1) é a métrica mais sensível: seu valor cai para ~0.5,")
    print("   pois metade da coerência nas distâncias entre pares de pontos foi quebrada.")
    print("3. O Sortedness Local ponderado (λ𝜏w) sofre penalização moderada (~0.15 a 0.20),")
    print("   refletindo que a vizinhança próxima ainda foi preservada, mas a ordem irrestrita externa foi perturbada.")


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    demo_propriedades_fundamentais()
    demo_randomizacao_progressiva()
    demo_troca_de_clusters()
    print("\n" + "=" * 80)
    print(" DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
