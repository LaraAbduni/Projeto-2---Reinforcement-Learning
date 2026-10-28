import json
from pathlib import Path
import random
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import get_data_path
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from QLearning import QLearning
from Sarsa import Sarsa


# Mesmos hiperparametros para os dois agentes e os dois ambientes.
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DEC = 0.995
EPISODES = 15000
SEEDS = [10, 20, 30, 40, 50]
EVALUATIONS = 100
RAINY_PROBABILITY = 0.6
WINDOW = 100

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
IMG = BASE.parent / 'img'
AMBIENTES = ['deterministico', 'nao-deterministico']
ALGORITMOS = {'QLearning': QLearning, 'Sarsa': Sarsa}
NOMES = {'QLearning': 'Q-Learning', 'Sarsa': 'Sarsa',
         'deterministico': 'Determinístico', 'nao-deterministico': 'Não determinístico'}


def criar_ambiente(ambiente, render_mode=None):
    return gym.make('Taxi-v4', is_rainy=ambiente == 'nao-deterministico',
                    rainy_probability=RAINY_PROBABILITY, render_mode=render_mode)


def gerar_animacao(algoritmo, ambiente, run=1, seed=10000):
    IMG.mkdir(exist_ok=True)
    q_table = np.loadtxt(DATA / 'q-table-Taxi-{}-{}-{}.csv'.format(algoritmo, ambiente, run), delimiter=',')
    env = criar_ambiente(ambiente, render_mode='rgb_array')
    (state, _) = env.reset(seed=seed)
    actions = 0
    rewards = 0
    terminated = False
    truncated = False
    escorregoes = 0
    frames = []
    nomes_acoes = ['baixo', 'cima', 'direita', 'esquerda', 'buscar', 'deixar']
    movimentos = {(1, 0): 'baixo', (-1, 0): 'cima', (0, 1): 'direita',
                  (0, -1): 'esquerda', (0, 0): 'ficou no lugar'}
    fonte = ImageFont.truetype(str(Path(get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'), size=16)

    def guardar_frame(acao, movimento):
        frame = Image.fromarray(env.render())
        imagem = Image.new('RGB', (frame.width, frame.height + 102), 'white')
        imagem.paste(frame, (0, 102))
        texto = ImageDraw.Draw(imagem)
        passageiro = tuple(env.unwrapped.decode(state))[2]
        situacao = 'Entregou o passageiro!' if terminated else (
            'Limite de 200 ações' if truncated else (
                'Passageiro no táxi' if passageiro == 4 else 'Indo buscar o passageiro'))
        texto.text((12, 8), '{} | {}'.format(NOMES[algoritmo], NOMES[ambiente]), font=fonte, fill='black')
        texto.text((12, 31), 'Passo: {} | Reward acumulado: {}'.format(actions, rewards), font=fonte, fill='black')
        texto.text((12, 54), 'Escolheu: {} | Movimento: {}'.format(acao, movimento), font=fonte, fill='black')
        texto.text((12, 77), situacao, font=fonte, fill='darkgreen' if terminated else 'black')
        frames.append(np.asarray(imagem))

    try:
        guardar_frame('início', '-')
        while not (terminated or truncated):
            row, col, _, _ = tuple(env.unwrapped.decode(state))
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            next_row, next_col, _, _ = tuple(env.unwrapped.decode(state))
            movimento = movimentos[(next_row - row, next_col - col)]
            if action < 4 and movimento != 'ficou no lugar' and movimento != nomes_acoes[action]:
                escorregoes = escorregoes + 1
            actions = actions + 1
            rewards = rewards + reward
            guardar_frame(nomes_acoes[action], movimento)
    finally:
        env.close()

    filename = IMG / 'taxi-{}-{}.gif'.format(algoritmo, ambiente)
    duracoes = [800] + [350] * (len(frames) - 2) + [1800]
    imageio.mimsave(filename, frames, format='GIF', duration=duracoes, loop=0)
    return filename, {'acoes': actions, 'reward': rewards, 'sucesso': bool(terminated),
                      'truncado': bool(truncated), 'escorregoes_visiveis': escorregoes}


def avaliar(q_table, algoritmo, treino, teste, run, seed):
    env = criar_ambiente(teste)
    resultados = []
    for ep in range(EVALUATIONS):
        # Mesmos estados iniciais e sementes de teste para todos os agentes.
        evaluation_seed = 10000 + ep
        (state, _) = env.reset(seed=evaluation_seed)
        done = False
        actions = 0
        rewards = 0
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            actions = actions + 1
            rewards = rewards + reward
            done = terminated or truncated
        resultados.append({'algoritmo': algoritmo, 'ambiente_treino': treino,
                           'ambiente_teste': teste, 'execucao': run, 'seed_treino': seed,
                           'episodio': ep + 1, 'seed_teste': evaluation_seed,
                           'acoes': actions, 'reward': rewards,
                           'sucesso': int(terminated), 'truncado': int(truncated)})
    env.close()
    return resultados


def executar_experimentos(algoritmo):
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    for ambiente in AMBIENTES:
        treinos = []
        avaliacoes = []
        for run, seed in enumerate(SEEDS, start=1):
            print('\n{} - {} - treinamento {}/{}'.format(algoritmo, ambiente, run, len(SEEDS)), flush=True)
            random.seed(seed)
            np.random.seed(seed)
            env = criar_ambiente(ambiente)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            agente = ALGORITMOS[algoritmo](env, alpha=ALPHA, gamma=GAMMA,
                     epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                     epsilon_dec=EPSILON_DEC, episodes=EPISODES)
            filename = DATA / 'q-table-Taxi-{}-{}-{}.csv'.format(algoritmo, ambiente, run)
            q_table = agente.train(filename, None)
            env.close()

            for ep in range(EPISODES):
                treinos.append({'algoritmo': algoritmo, 'ambiente_treino': ambiente,
                                'execucao': run, 'seed': seed, 'episodio': ep + 1,
                                'reward': agente.rewards_per_episode[ep],
                                'acoes': agente.actions_per_episode[ep]})

            # Sem chuva -> sem chuva e com chuva; com chuva -> com chuva.
            testes = AMBIENTES if ambiente == 'deterministico' else ['nao-deterministico']
            for teste in testes:
                avaliacoes.extend(avaliar(q_table, algoritmo, ambiente, teste, run, seed))

        pd.DataFrame(treinos).to_csv(RESULTS / 'treinamento-{}-{}.csv'.format(algoritmo, ambiente), index=False)
        pd.DataFrame(avaliacoes).to_csv(RESULTS / 'avaliacao-{}-{}.csv'.format(algoritmo, ambiente), index=False)


def carregar_resultados():
    treinos = []
    avaliacoes = []
    for algoritmo in ALGORITMOS:
        for ambiente in AMBIENTES:
            treinos.append(pd.read_csv(RESULTS / 'treinamento-{}-{}.csv'.format(algoritmo, ambiente)))
            avaliacoes.append(pd.read_csv(RESULTS / 'avaliacao-{}-{}.csv'.format(algoritmo, ambiente)))
    return pd.concat(treinos, ignore_index=True), pd.concat(avaliacoes, ignore_index=True)


def resumir_avaliacoes(avaliacoes):
    grupos = ['algoritmo', 'ambiente_treino', 'ambiente_teste', 'execucao']
    por_execucao = avaliacoes.groupby(grupos, sort=False).agg(
        episodios=('acoes', 'size'), media_acoes=('acoes', 'mean'),
        desvio_padrao_acoes=('acoes', lambda x: np.std(x, ddof=0)),
        sucessos=('sucesso', 'sum'), truncados=('truncado', 'sum'),
        reward_medio=('reward', 'mean')).reset_index()
    geral = avaliacoes.groupby(grupos[:-1], sort=False).agg(
        episodios=('acoes', 'size'), media_acoes=('acoes', 'mean'),
        desvio_padrao_acoes=('acoes', lambda x: np.std(x, ddof=0)),
        sucessos=('sucesso', 'sum'), truncados=('truncado', 'sum'),
        reward_medio=('reward', 'mean')).reset_index()
    # Separar conclusoes evita chamar um episodio truncado de tarefa concluida.
    sucessos = avaliacoes[avaliacoes['sucesso'] == 1].groupby(grupos[:-1], sort=False).agg(
        media_acoes_sucessos=('acoes', 'mean'),
        desvio_padrao_acoes_sucessos=('acoes', lambda x: np.std(x, ddof=0))).reset_index()
    geral = geral.merge(sucessos, on=grupos[:-1], how='left')
    geral['taxa_sucesso'] = 100 * geral['sucessos'] / geral['episodios']
    return por_execucao, geral


def plotar_aprendizado(treinos, ambiente):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for algoritmo, cor in [('QLearning', 'tab:blue'), ('Sarsa', 'tab:orange')]:
        dados = treinos[(treinos['algoritmo'] == algoritmo) & (treinos['ambiente_treino'] == ambiente)]
        rewards = dados.pivot(index='episodio', columns='execucao', values='reward')
        curvas = rewards.rolling(WINDOW, min_periods=WINDOW).mean()
        for ax, detalhe in zip(axes, [False, True]):
            dados_plot = curvas[curvas.index > curvas.index.max() - 5000] if detalhe else curvas
            for run in dados_plot.columns:
                ax.plot(dados_plot.index, dados_plot[run], color=cor, alpha=0.15, linewidth=0.6)
            media = dados_plot.mean(axis=1)
            desvio = dados_plot.std(axis=1, ddof=0)
            ax.plot(dados_plot.index, media, color=cor, label=NOMES[algoritmo])
            ax.fill_between(dados_plot.index, media - desvio, media + desvio, color=cor, alpha=0.15)
    meta = gym.spec('Taxi-v4').reward_threshold
    for ax, titulo in zip(axes, ['Treinamento completo', 'Últimos 5.000 episódios']):
        ax.axhline(meta, color='black', linestyle='--', label='Referência determinística: reward médio = {}'.format(meta))
        ax.set(xlabel='Episódios', ylabel='Reward acumulado (média móvel de 100 episódios)', title=titulo)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle('Curvas de aprendizado - {}'.format(NOMES[ambiente]))
    fig.tight_layout()
    fig.savefig(RESULTS / 'aprendizado-{}.jpg'.format(ambiente), dpi=160)
    return fig


def plotar_acoes(resumo, treino, teste):
    dados = resumo[(resumo['ambiente_treino'] == treino) & (resumo['ambiente_teste'] == teste)]
    dados = dados.set_index('algoritmo').loc[['QLearning', 'Sarsa']]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([NOMES[x] for x in dados.index], dados['media_acoes'],
                  yerr=dados['desvio_padrao_acoes'], capsize=8,
                  color=['tab:blue', 'tab:orange'], alpha=0.8)
    limite = (dados['media_acoes'] + dados['desvio_padrao_acoes']).max()
    for bar, (_, row) in zip(bars, dados.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                row['media_acoes'] + row['desvio_padrao_acoes'] + limite * 0.03,
                '{:.2f} ± {:.2f}\nSucesso: {:.1f}%'.format(
                    row['media_acoes'], row['desvio_padrao_acoes'], row['taxa_sucesso']),
                ha='center', fontsize=9)
    ax.set_ylim(0, limite * 1.30)
    ax.set(ylabel='Quantidade de ações por episódio',
           title='Treino: {}\nTeste: {}'.format(NOMES[treino], NOMES[teste]))
    ax.text(0.5, -0.13, 'Média ± desvio padrão (ddof=0); 5 Q-tables × 100 testes.\nInclui episódios truncados no limite de 200 ações.',
            transform=ax.transAxes, ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'acoes-{}-para-{}.jpg'.format(treino, teste), dpi=160)
    return fig


def plotar_caso(treinos, por_execucao, algoritmo, ambiente):
    dados = treinos[(treinos['algoritmo'] == algoritmo) &
                    (treinos['ambiente_treino'] == ambiente)]
    testes = por_execucao[(por_execucao['algoritmo'] == algoritmo) &
                          (por_execucao['ambiente_treino'] == ambiente) &
                          (por_execucao['ambiente_teste'] == ambiente)]
    cor = 'tab:blue' if algoritmo == 'QLearning' else 'tab:orange'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    rewards = dados.pivot(index='episodio', columns='execucao', values='reward')
    curvas = rewards.rolling(WINDOW, min_periods=WINDOW).mean()
    for run in curvas.columns:
        axes[0].plot(curvas.index, curvas[run], color=cor, alpha=0.2, linewidth=0.7)
    axes[0].plot(curvas.index, curvas.mean(axis=1), color=cor, label='Média dos 5 treinos')
    axes[0].axhline(gym.spec('Taxi-v4').reward_threshold, color='black', linestyle='--',
                   label='Referência sem chuva: 8')
    axes[0].set(xlabel='Episódios', ylabel='Reward (média móvel de 100 episódios)',
                title='Aprendizado')
    axes[0].legend(fontsize=8)

    axes[1].bar(testes['execucao'], testes['media_acoes'],
                yerr=testes['desvio_padrao_acoes'], capsize=6, color=cor, alpha=0.8)
    axes[1].set(xlabel='Q-table avaliada (100 testes por execução)',
                ylabel='Quantidade de ações', title='Média e desvio padrão das ações')
    axes[1].set_xticks(testes['execucao'])
    for ax in axes:
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('{} no ambiente {}'.format(NOMES[algoritmo], NOMES[ambiente].lower()))
    fig.tight_layout()
    fig.savefig(RESULTS / 'caso-{}-{}.jpg'.format(algoritmo, ambiente), dpi=160)
    return fig


def gerar_resultados():
    treinos, avaliacoes = carregar_resultados()
    por_execucao, resumo = resumir_avaliacoes(avaliacoes)
    por_execucao.to_csv(RESULTS / 'resumo-por-execucao.csv', index=False)
    resumo.to_csv(RESULTS / 'resumo-geral.csv', index=False)
    for algoritmo in ALGORITMOS:
        for ambiente in AMBIENTES:
            plt.close(plotar_caso(treinos, por_execucao, algoritmo, ambiente))
    for ambiente in AMBIENTES:
        plt.close(plotar_aprendizado(treinos, ambiente))
    for treino, teste in [('deterministico', 'deterministico'),
                          ('deterministico', 'nao-deterministico'),
                          ('nao-deterministico', 'nao-deterministico')]:
        plt.close(plotar_acoes(resumo, treino, teste))
    config = {'ambiente': 'Taxi-v4', 'gymnasium': gym.__version__,
              'alpha': ALPHA, 'gamma': GAMMA, 'epsilon': EPSILON,
              'epsilon_min': EPSILON_MIN, 'epsilon_dec': EPSILON_DEC,
              'episodios': EPISODES, 'seeds': SEEDS,
              'avaliacoes_por_q_table': EVALUATIONS, 'rainy_probability': RAINY_PROBABILITY,
              'limite_acoes': gym.spec('Taxi-v4').max_episode_steps,
              'meta_reward_deterministico': gym.spec('Taxi-v4').reward_threshold,
              'janela_media_movel': WINDOW, 'desvio_padrao_ddof': 0,
              'numpy': np.__version__, 'pandas': pd.__version__}
    (RESULTS / 'configuracao.json').write_text(json.dumps(config, indent=4), encoding='utf-8')
    print('\n' + resumo.to_string(index=False))
    return resumo


if __name__ == '__main__':
    for algoritmo in ALGORITMOS:
        executar_experimentos(algoritmo)
    gerar_resultados()
