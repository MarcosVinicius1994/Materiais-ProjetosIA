#!/usr/bin/env python3
# -*- codificacao: utf-8 -*-
"""
Created on Sun Sep 23 15:33:59 2018
@author: talles medeiros, decsi-ufop
"""

"""
Este código servirá de exemplo para o aprendizado do algoritmo MINIMAX 
na disciplina de Inteligência Artificial - CSI457
Semestre: 2018/2
"""

#!/usr/bin/env python3
from math import inf as infinity
from random import choice
import platform
import time
from os import system

"""
Um versão simples do algoritmo MINIMAX para o Jogo da Velha.
"""

# Representando a variável que identifica cada jogador
# HUMANO = Oponente humano
# COMP = Agente Inteligente
# tabuleiro = dicionário com os valores em cada posição (x,y)
# indicando o jogador que movimentou nessa posição.
# Começa vazio, com zero em todas posições.
HUMANO = -1
COMP = +1
tabuleiro = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
"""
Funcao para avaliacao heuristica do estado.
:parametro (estado): o estado atual do tabuleiro
:returna: +1 se o computador vence; -1 se o HUMANOo vence; 0 empate
 """
def avaliacao(estado):
    
    if vitoria(estado, COMP):
        placar = +1
    elif vitoria(estado, HUMANO):
        placar = -1
    else:
        placar = 0

    return placar
""" fim avaliacao (estado)------------------------------------- """

def vitoria(estado, jogador):
    """
    Esta funcao testa se um jogador especifico vence. Possibilidades:
    * Tres linhas     [X X X] or [O O O]
    * Tres colunas    [X X X] or [O O O]
    * Duas diagonais  [X X X] or [O O O]
    :param. (estado): o estado atual do tabuleiro
    :param. (jogador): um HUMANO ou um Computador
    :return: True se jogador vence
    """
    win_estado = [
        [estado[0][0], estado[0][1], estado[0][2], estado[0][3]], # toda linha 1
        [estado[1][0], estado[1][1], estado[1][2], estado[1][3]], # toda linha 2
        [estado[2][0], estado[2][1], estado[2][2], estado[2][3]], # toda linha 3
        [estado[3][0], estado[3][1], estado[3][2], estado[3][3]], # toda linha 4
        [estado[0][0], estado[1][0], estado[2][0], estado[3][0]], # toda coluna 1
        [estado[0][1], estado[1][1], estado[2][1], estado[3][1]], # toda coluna 2
        [estado[0][2], estado[1][2], estado[2][2], estado[3][2]], # toda coluna 3
        [estado[0][3], estado[1][3], estado[2][3], estado[3][3]], # toda coluna 4
        [estado[0][0], estado[1][1], estado[2][2], estado[3][3]], # diagonal principal
        [estado[3][0], estado[2][1], estado[1][2], estado[0][3]], # diagonal secundária
    ]
    # Se um, dentre todos os alinhamentos pertence um mesmo jogador, 
    # então o jogador vence!
    if [jogador, jogador, jogador, jogador] in win_estado:
        return True
    else:
        return False
""" ---------------------------------------------------------- """
def getBoardCopy(estado):
    #Faz uma copia do quadro e retrona esta copia

    dupeBoard = []

    for i in estado:
        dupeBoard.append(i)

    return dupeBoard

"""
Testa fim de jogo para ambos jogadores de acordo com estado atual
return: será fim de jogo caso ocorra vitória de um dos jogadores.
"""
def fim_jogo(estado):
    return vitoria(estado, HUMANO) or vitoria(estado, COMP)
""" ---------------------------------------------------------- """

"""
Verifica celular vazias e insere na lista para informar posições
ainda permitidas para próximas jogadas.
"""
def celulas_vazias(estado):
    celulas = []
    for x, row in enumerate(estado):
        for y, cell in enumerate(row):
            if cell == 0: celulas.append([x, y])
    return celulas
""" ---------------------------------------------------------- """

"""
Um movimento é valido se a célula escolhida está vazia.
:param (x): coordenada X
:param (y): coordenada Y
:return: True se o tabuleiro[x][y] está vazio
"""
def movimento_valido(x, y):
    if [x, y] in celulas_vazias(tabuleiro):
        return True
    else:
        return False
""" ---------------------------------------------------------- """

"""
Executa o movimento no tabuleiro se as coordenadas são válidas
:param (x): coordenadas X
:param (y): coordenadas Y
:param (jogador): o jogador da vez
"""
def exec_movimento(x, y, jogador):
    if movimento_valido(x, y):
        tabuleiro[x][y] = jogador
        return True
    else:
        return False
""" ---------------------------------------------------------- """

"""
Função da IA que escolhe o melhor movimento
:param (estado): estado atual do tabuleiro
:param (profundidade): índice do nó na árvore (0 <= profundidade <= 9),
mas nunca será nove neste caso (veja a função iavez())
:param (jogador): um HUMANO ou um Computador
:return: uma lista com [melhor linha, melhor coluna, melhor placar]
"""
def minimax(estado, profundidade, jogador, alfa, beta): #print('Profundidade inicial: ',profundidade)  input("Pressione <enter> para continuar")

    # valor-minmax(estado)
    if jogador == COMP:
        melhor = [-1, -1, -infinity] #print('estado mim-max ', melhor)
    else:
        melhor = [-1, -1, +infinity]

    # valor-minimax(estado) = avaliacao(estado)
    if profundidade == 0 or fim_jogo(estado):   #print('Profundidade inicial minmax: ',profundidade)  input("Pressione <enter> para continuar")
        placar = avaliacao(estado, profundidade)    # print('estado de avaliação ',placar)
        return [-1, -1, placar]

    for cell in celulas_vazias(estado):
        x, y = cell[0], cell[1]
        estado[x][y] = jogador      #print('minMax celula ', estado)
        placar = minimax(estado, profundidade - 1, -jogador, alfa, beta)
        estado[x][y] = 0
        placar[0], placar[1] = x, y     #print('Profundidade: ',profundidade)

        if jogador == COMP:
            if placar[2] > melhor[2]:
                melhor = placar # valor MAX
                alfa = placar[2]
            if alfa >= beta:
                break
        else:
            if placar[2] < melhor[2]:
                melhor = placar  # valor MIN
                beta = placar[2]
            if alfa >= beta:
                break

    return melhor
""" ---------------------------------------------------------- """
"""
Limpa o console para SO Windows
"""
def limpa_console():
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')
""" ---------------------------------------------------------- """

"""
Imprime o tabuleiro no console
:param. (estado): estado atual do tabuleiro
"""
def exibe_tabuleiro(estado, comp_escolha, humano_escolha):
    print('---------------------')
    for row in estado:
        print('\n---------------------')
        for cell in row:
            if cell == +1:
                print('|', comp_escolha, '|', end='')
            elif cell == -1:
                print('|', humano_escolha, '|', end='')
            else:
                print('|', ' ', '|', end='')
    print('\n----------------------')
""" ---------------------------------------------------------- """

"""
Chama a função minimax se a profundidade < 9,
ou escolhe uma coordenada aleatória.
:param (comp_escolha): Computador escolhe X ou O
:param (humano_escolha): HUMANO escolhe X ou O
:return:
"""
def IA_vez(comp_escolha, humano_escolha):
    profundidade = len(celulas_vazias(tabuleiro))   #print('profundidade: ',profundidade) input("Pressione <enter> para continuar")
    if profundidade == 0 or fim_jogo(tabuleiro):
        return

    limpa_console()
    print('Vez do Computador [{}]'.format(comp_escolha))
    exibe_tabuleiro(tabuleiro, comp_escolha, humano_escolha)

    if profundidade == 16:
        x = choice([0, 1, 2, 3])
        y = choice([0, 1, 2, 3])
    else:
        move = minimax(tabuleiro, profundidade, COMP, -infinity, +infinity)
        x, y = move[0], move[1]

    exec_movimento(x, y, COMP)
    time.sleep(1)
""" ---------------------------------------------------------- """
def HUMANO_vez(comp_escolha, humano_escolha):
    """
    O HUMANO joga escolhendo um movimento válido
    :param comp_escolha: Computador escolhe X ou O
    :param humano_escolha: HUMANO escolhe X ou O
    :return:
    """
    profundidade = len(celulas_vazias(tabuleiro))
    if profundidade == 0 or fim_jogo(tabuleiro):
        return

    # Dicionário de movimentos válidos
    movimento = -1
    movimentos = {
        1: [0, 0], 2: [0, 1], 3: [0, 2], 4:[0, 3],
        5: [1, 0], 6: [1, 1], 7: [1, 2], 8:[1, 3],
        9: [2, 0], 10: [2, 1], 11: [2, 2], 12: [2, 3],
        13:[3, 0], 14: [3, 1], 15: [3, 2], 16: [3, 3]
    }

    limpa_console()
    print('Vez do HUMANO [{}]'.format(humano_escolha))
    exibe_tabuleiro(tabuleiro, comp_escolha, humano_escolha)

    while (movimento < 1 or movimento > 16):
        try:
            movimento = int(input('Use numero (1..16): '))
            coord = movimentos[movimento]
            tenta_movimento = exec_movimento(coord[0], coord[1], HUMANO)

            if tenta_movimento == False:
                print('Movimento Inválido')
                xmovimento = -1
        except KeyboardInterrupt:
            print('Tchau!')
            exit()
        except:
            print('Escolha Inválida!')
""" ---------------------------------------------------------- """

""" Função para fazer a poda Alpha/Beta que não foi utilizada"""
# def alphabeta(tabuleiro, comp, turn, alpha, beta):
"""if COMP == 'X':
		HUMANO = 'O'
	else:
		HUMANO = 'X'
		
	if turn == COMP:
		nextTurn = HUMANO
	else:
		nextTurn = COMP
		
	finish = avaliacao(tabuleiro)
	
		possiveis = celulas_vazias(tabuleiro)
		
		if turn == COMP:
		
		for move in possiveis:
		    tabuleiro[move] = turn
		    val = alphabeta(tabuleiro, COMP, nextTurn, alpha, beta)
		    tabuleiro[move] = ''
		    
		    if val > alpha:
				alpha = val

			if alpha >= beta:
				return alpha
		return alpha
		
		else:
		for move in possiveis:
		       tabuleiro[move] = turn
		       val = alphabeta(tabuleiro, COMP, nextTurn, alpha, beta)
		       tabuleiro[move] = ''
		     if val > beta:
		         beta = val
		         
		     if alpha >= beta:
				return beta
		return beta
					
"""


"""
Funcao Principal que chama todas funcoes
"""
def main():

    limpa_console()
    humano_escolha = '' # Pode ser X ou O
    comp_escolha = '' # Pode ser X ou O
    primeiro = ''  # se HUMANO e o primeiro

    # HUMANO escolhe X ou O para jogar
    while humano_escolha != 'O' and humano_escolha != 'X':
        try:
            print('')
            humano_escolha = input('Escolha X or O\n: ').upper()
        except KeyboardInterrupt:
            print('Tchau!')
            exit()
        except:
            print('Escolha Errada')

    # Setting Computador's choice
    if humano_escolha == 'X':
        comp_escolha = 'O'
    else:
        comp_escolha = 'X'

    # HUMANO pode começar primeiro
    limpa_console()
    while primeiro != 'S' and primeiro != 'N':
        try:
            primeiro = input('Primeiro a Iniciar?[s/n]: ').upper()
        except KeyboardInterrupt:
            print('Tchau!')
            exit()
        except:
            print('Escolha Errada!')

    # Laço principal do jogo
    while len(celulas_vazias(tabuleiro)) > 0 and not fim_jogo(tabuleiro):
        if primeiro == 'N':
            IA_vez(comp_escolha, humano_escolha)
            primeiro = ''

        HUMANO_vez(comp_escolha, humano_escolha)
        IA_vez(comp_escolha, humano_escolha)

    # Mensagem de Final de jogo
    if vitoria(tabuleiro, HUMANO):
        limpa_console()
        print('Vez do HUMANO [{}]'.format(humano_escolha))
        exibe_tabuleiro(tabuleiro, comp_escolha, humano_escolha)
        print('Você Venceu!')
    elif vitoria(tabuleiro, COMP):
        limpa_console()
        print('Vez do Computador [{}]'.format(comp_escolha))
        exibe_tabuleiro(tabuleiro, comp_escolha, humano_escolha)
        print('Você Perdeu!')
    else:
        limpa_console()
        exibe_tabuleiro(tabuleiro, comp_escolha, humano_escolha)
        print('Empate!')

    exit()

if __name__ == '__main__':
    main()