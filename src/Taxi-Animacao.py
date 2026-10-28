from TaxiDriverGym import ALGORITMOS, AMBIENTES, gerar_animacao


# Usa a primeira Q-table de cada caso, sem treinar novamente.
if __name__ == '__main__':
    for algoritmo in ALGORITMOS:
        for ambiente in AMBIENTES:
            arquivo, resultado = gerar_animacao(algoritmo, ambiente)
            print(arquivo.name)
            print('Ações: {} | Reward: {} | Entregou: {}'.format(
                resultado['acoes'], resultado['reward'], resultado['sucesso']))
