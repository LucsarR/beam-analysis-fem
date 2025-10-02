# Lucas Reinaldo Sarmento e Matheus da Silva Leal

import numpy as np

# Leitura do arquivo de entrada
arquivo_entrada = 'entrada_Teste2.txt'  # SUBSTITUA AQUI
with open(arquivo_entrada, 'r') as file:
    lines = file.readlines()

# Inicialização das estruturas
points = []
curves = []
materials = []
properties = []
mesh = []
point_loads = []
dist_loads = []
bc = []

# Processamento das linhas
section = None
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith('*'):
        section = line.split()[0][1:].lower()
        continue
    eval(section).append(line.split(','))

# Pré-processamento dos dados
nos = []
elementos = []
id_aux_no = len(points) + 1
id_aux_elem = 1

# Subdivisão da malha
for malha in mesh:
    id_curva = int(malha[0])
    id_propriedade = int(malha[1])
    num_elementos = int(malha[2])

    # Extração da curva
    for curva in curves:
        if int(curva[0]) == id_curva:
            id_no_ini = int(curva[1])
            id_no_fim = int(curva[2])
            break

    # Extração das propriedades
    for prop in properties:
        if int(prop[0]) == id_propriedade:
            id_material = int(prop[1])
            for mat in materials:
                if int(mat[0]) == id_material:
                    E = float(mat[1])
                    nu = float(mat[2])
                    break
            A = float(prop[2])
            I = float(prop[3])
            break
    
    # Extração dos nós
    for ponto in points:
        if int(ponto[0]) == id_no_ini:
            x1, y1 = float(ponto[1]), float(ponto[2])
        elif int(ponto[0]) == id_no_fim:
            x2, y2 = float(ponto[1]), float(ponto[2])

    dx = (x2 - x1) / num_elementos
    dy = (y2 - y1) / num_elementos
    
    # Extração das cargas distribuídas (q_ini, q_fim, p_ini, p_fim)
    cargas_distribuidas = np.zeros(4)
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    for carga in dist_loads:
        if int(carga[1]) == id_curva:
            if str(carga[4]) == 'x':
                cargas_distribuidas[0] += float(carga[2]) * c
                cargas_distribuidas[1] += float(carga[3]) * c
                cargas_distribuidas[2] += - float(carga[2]) * s
                cargas_distribuidas[3] += - float(carga[3]) * s
            if str(carga[4]) == 'y':
                cargas_distribuidas[0] += float(carga[2]) * s
                cargas_distribuidas[1] += float(carga[3]) * s
                cargas_distribuidas[2] += float(carga[2]) * c
                cargas_distribuidas[3] += float(carga[3]) * c
            if str(carga[4]) == 'l':
                cargas_distribuidas[0] += float(carga[2])
                cargas_distribuidas[1] += float(carga[3])
            if str(carga[4]) == 't':
                cargas_distribuidas[2] += float(carga[2])
                cargas_distribuidas[3] += float(carga[3])

    # Criação dos nós
    if [int(id_no_ini), float(x1), float(y1)] not in nos:
        nos.append([int(id_no_ini), float(x1), float(y1)])
    for i in range(1, num_elementos):
        id_novo_no = id_aux_no
        id_aux_no += 1
        novo_x = x1 + i * dx
        novo_y = y1 + i * dy
        nos.append([int(id_novo_no), float(novo_x), float(novo_y)])
    if [int(id_no_fim), float(x2), float(y2)] not in nos:
        nos.append([int(id_no_fim), float(x2), float(y2)])
    
    # Criação dos elementos
    delta_q = (cargas_distribuidas[1] - cargas_distribuidas[0]) / num_elementos
    delta_p = (cargas_distribuidas[3] - cargas_distribuidas[2]) / num_elementos
    q_aux = cargas_distribuidas[0]
    p_aux = cargas_distribuidas[2]
    for i in range(num_elementos):
        novo_no_ini = id_no_ini if i == 0 else id_aux_no - num_elementos + i
        novo_no_fim = id_no_fim if i == num_elementos - 1 else id_aux_no - num_elementos + i + 1
        cargas_distribuidas = [q_aux, q_aux + delta_q, p_aux, p_aux + delta_p]
        q_aux += delta_q
        p_aux += delta_p

        elementos.append([id_aux_elem , novo_no_ini, novo_no_fim, E, A, I, nu, cargas_distribuidas])
        id_aux_elem += 1


# Função para calcular L, c e s de um elemento
def calcular_L_c_s(x1, y1, x2, y2):
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    return L, c, s

# Função para calcular Ke de um elemento
def calcular_ke(L, c, s, E, A, I):
    mu = (A * L**2) / (2 * I)
    aux_1 = mu * c**2 + 6 * s**2
    aux_2 = mu * s**2 + 6 * c**2
    aux_3 = (mu - 6) * c * s
    aux_4 = 3 * L * c
    aux_5 = 3 * L * s
    aux_6 = L**2
    k_local = 2 * E * I / L**3 * np.array([[aux_1, aux_3, -aux_5, -(aux_1), -aux_3, -aux_5],
                                           [aux_3, (aux_2), aux_4, -aux_3, -(aux_2), aux_4],
                                           [-aux_5, aux_4, 2*aux_6, aux_5, -aux_4, aux_6],
                                           [-(aux_1), -aux_3, aux_5, aux_1, aux_3, aux_5],
                                           [-aux_3, -(aux_2), -aux_4, aux_3, aux_2, -aux_4],
                                           [-aux_5, aux_4, aux_6, aux_5, -aux_4, 2*aux_6]])
    return k_local

# Função para calcular fe de um elemento
def calcular_fe(L, c, s, q_ini, q_fim, p_ini, p_fim):
    R = np.array([[c, -s, 0, 0, 0, 0],
                  [s, c, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c, -s, 0],
                  [0, 0, 0, s, c, 0],
                  [0, 0, 0, 0, 0, 1]])
    fe_local = L * R @ np.array([[(2*q_ini + q_fim) / 6],
                                 [(7*p_ini + 3*p_fim) / 20],
                                 [(3*p_ini + 2*p_fim) * L / 60],
                                 [(q_ini + 2*q_fim) / 6],
                                 [(3*p_ini + 7*p_fim) / 20],
                                 [- (2*p_ini + 3*p_fim) * L / 60]])
    return fe_local

# Cálculo de Ke e fe para cada elemento
ke_lista = []
fe_lista = []

# Função para encontrar as coordenadas de um nó
def x_y_no(id_no):
    for no in nos:
        if int(no[0]) == id_no:
            return float(no[1]), float(no[2])

for elemento in elementos:

    id_no1 = int(elemento[1])
    id_no2 = int(elemento[2])
    E = float(elemento[3])
    A = float(elemento[4])
    I = float(elemento[5])
    nu = float(elemento[6])
    q_ini = float(elemento[7][0])
    q_fim = float(elemento[7][1])
    p_ini = float(elemento[7][2])
    p_fim = float(elemento[7][3])

    x1, y1 = x_y_no(id_no1)
    x2, y2 = x_y_no(id_no2)
    L, c, s = calcular_L_c_s(x1, y1, x2, y2)
    
    ke = calcular_ke(L, c, s, E, A, I)
    fe = calcular_fe(L, c, s, q_ini, q_fim, p_ini, p_fim)
    
    ke_lista.append(ke)
    fe_lista.append(fe)


num_nos = len(nos)
Kg = np.zeros((3*num_nos, 3*num_nos))
fg = np.zeros((3*num_nos, 1))

# Montagem de Kg e fg
for id_elemento, elemento in enumerate(elementos):

    id_no1 = int(elemento[1])
    id_no2 = int(elemento[2])

    id_global = [   # Índices globais dos nós na matriz de rigidez global
        3*(id_no1-1), 3*(id_no1-1)+1, 3*(id_no1-1)+2,  # Posição global do nó 1
        3*(id_no2-1), 3*(id_no2-1)+1, 3*(id_no2-1)+2   # Posição global do nó 2
    ]

    ke = ke_lista[id_elemento]
    fe = fe_lista[id_elemento]

    for i in range(6):
        for j in range(6):
            Kg[id_global[i], id_global[j]] += ke[i, j]

    for i in range(6):
        fg[id_global[i]] += fe[i]

# Somando as forças externas
for carga in point_loads:
    id_no = int(carga[1])
    gdl = int(carga[2])
    valor = float(carga[3])

    if gdl == 1: # Força em x
        fg[3*(id_no-1)] += valor
    elif gdl == 2: # Força em y
        fg[3*(id_no-1)+1] += valor
    elif gdl == 3: # Momento
        fg[3*(id_no-1)+2] += valor


# Aplicação das condições de contorno
def aplicar_condicoes_contorno(Kcc, fcc, id_cc, valor):

    Kcc[id_cc, :] = 0
    if valor == 0:
        Kcc[:, id_cc] = 0
    Kcc[id_cc, id_cc] = 1
    fcc[id_cc] = valor

    return Kcc, fcc

Kcc = np.copy(Kg)
fcc = np.copy(fg)

for cc in bc:
    id_no = int(cc[0])
    gdl = int(cc[1])
    valor = float(cc[2])

    if gdl == 1: # Condição de contorno em x
        id_cc = 3*(id_no-1)
    elif gdl == 2: # Condição de contorno em y
        id_cc = 3*(id_no-1)+1
    elif gdl == 3: # Condição de contorno em momento
        id_cc = 3*(id_no-1)+2
    Kcc, fcc = aplicar_condicoes_contorno(Kcc, fcc, id_cc, valor)


# Pré-processamento para resolver o sistema linear
# Para evitar o erro de singularidade, aplica-se condições de contorno essenciais
# em linhas e colunas nulas.
for i in range(len(Kcc)):
    if np.all(Kcc[i] == 0):
        Kcc[i, i] = 1

# Resolver o sistema linear
ug = np.linalg.solve(Kcc, fcc)

# Calcular as reações de apoio
r = np.dot(Kg, ug) - fg

# Calcular o esforço normal, cortante e momento fletor
esforco_normal_lista = []
esforco_cortante_lista = []
momento_fletor_1_lista = []
momento_fletor_2_lista = []

for elemento in elementos:

    id_no1 = int(elemento[1])
    id_no2 = int(elemento[2])
    E = float(elemento[3])
    A = float(elemento[4])
    I = float(elemento[5])
    
    x1, y1 = x_y_no(id_no1)
    x2, y2 = x_y_no(id_no2)
    
    L, c, s = calcular_L_c_s(x1, y1, x2, y2)
    
    # Deslocamentos globais dos nós do elemento
    id_global = [   # Índices globais dos nós na matriz de rigidez global
        3*(id_no1-1), 3*(id_no1-1)+1, 3*(id_no1-1)+2,  # Posição global do nó 1
        3*(id_no2-1), 3*(id_no2-1)+1, 3*(id_no2-1)+2   # Posição global do nó 2
    ]

    # Parametro nodais nas coordenadas do elemento
    parametro_nodais = np.array([ug[id_global[0]], ug[id_global[1]], ug[id_global[2]], ug[id_global[3]], ug[id_global[4]], ug[id_global[5]]]).flatten()
    parametro_nodais_elemento = np.array([[c, s, 0, 0, 0, 0],
                                          [-s, c, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, c, s, 0],
                                          [0, 0, 0, -s, c, 0],
                                          [0, 0, 0, 0, 0, 1]]) @ parametro_nodais
    
    # Esforço normal
    esforco_normal = E * A * (1/L) * (parametro_nodais_elemento[3] - parametro_nodais_elemento[0])
    esforco_normal_lista.append(esforco_normal)

    # Esforço cortante
    esforco_cortante = E * I * (1/L**2) * (12*parametro_nodais_elemento[1]/L + 6*parametro_nodais_elemento[2] - 12*parametro_nodais_elemento[4]/L + 6*parametro_nodais_elemento[5])
    esforco_cortante_lista.append(esforco_cortante)

    # Momento fletor em relação ao nó 1
    momento_fletor_1 = E * I * (1/L) * (-6*parametro_nodais_elemento[1]/L - 4*parametro_nodais_elemento[2] + 6*parametro_nodais_elemento[4]/L - 2*parametro_nodais_elemento[5])
    momento_fletor_1_lista.append(momento_fletor_1)

    # Momento fletor em relação ao nó 2
    momento_fletor_2 = E * I * (1/L) * (6*parametro_nodais_elemento[1]/L + 2*parametro_nodais_elemento[2] - 6*parametro_nodais_elemento[4]/L + 4*parametro_nodais_elemento[5])
    momento_fletor_2_lista.append(momento_fletor_2)



# Print dos resultados no arquivo de saida
resultado = f"Resultado da Simulação\n\nArquivo de Entrada: {arquivo_entrada}\n\n\n"
resultado += "------------------ Resultados Nodais ------------------\n"
resultado += "|   Nó|              u|              v|          theta|\n"
resultado += "-------------------------------------------------------\n"
for no in nos:
    id_no = int(no[0])
    x = float(no[1])
    y = float(no[2])
    u = ug[3*(id_no-1), 0]
    v = ug[3*(id_no-1)+1, 0]
    theta = ug[3*(id_no-1)+2, 0]
    resultado += f"| {id_no:>4}| {u:>14.4f}| {v:>14.4f}| {theta:>14.6f}|\n"
resultado += "-------------------------------------------------------\n\n\n"

# Resultados por Elemento
resultado += "------------------------------------------------------ Resultados por Elemento ------------------------------------------------------\n"
resultado += "| Elemento|  Nó Inicial|        Força x|        Força y|        Momento|    Nó Final|        Força x|        Força y|        Momento|\n"
resultado += "-------------------------------------------------------------------------------------------------------------------------------------\n"
for elemento in elementos:
    id_elem = int(elemento[0])
    id_no_ini = int(elemento[1])
    id_no_fim = int(elemento[2])
    esforco_normal = esforco_normal_lista[id_elem-1]
    esforco_cortante = esforco_cortante_lista[id_elem-1]
    momento_fletor_1 = momento_fletor_1_lista[id_elem-1]
    momento_fletor_2 = momento_fletor_2_lista[id_elem-1]

    resultado += (f"| {id_elem:>8}| {id_no_ini:>11}| {float(esforco_normal):>14.4f}| {float(esforco_cortante):>14.4f}| {float(momento_fletor_1):>14.4f}|"
                                 f" {id_no_fim:>11}| {float(esforco_normal):>14.4f}| {float(esforco_cortante):>14.4f}| {float(momento_fletor_2):>14.4f}|\n")
resultado += "-------------------------------------------------------------------------------------------------------------------------------------\n\n\n"

# Forças de Reação
resultado += "------ Forças de Reação ------\n"
resultado += "|   Nó|   GDL|          Value|\n"
resultado += "------------------------------\n"
for reacao in bc:
    id_no = int(reacao[0])
    gdl = int(reacao[1])
    valor = r[3*(id_no-1)+gdl-1, 0]
    resultado += f"| {id_no:>4}| {gdl:>5}| {valor:>14.4f}|\n"
resultado += "------------------------------\n"

# saida do arquivo no formato correto e.g. se a entrada for entrada_Ex_Exercicio_Teste.txt => saida_Ex_Exercicio_Teste.txt
arquivo_saida = arquivo_entrada.replace('entrada', 'saida')
with open(arquivo_saida, 'w', encoding='utf-8') as file:
    file.write(resultado)