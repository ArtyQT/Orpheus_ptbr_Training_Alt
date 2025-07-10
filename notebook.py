'''Original notebook from freds0's Collab used to perform inference and/or training on Orpheus TTS model. Inference is performed on his GGUF (quantized) fine-tune (as per inference.py), whereas training is done on top of an Unsloth checkpoint. Modification requirements include: adding scripts responsible for registering user input and generating responses by setting the text to be passed to the inference srcipt; preferably, register user input by means of an STT model; linking training and inference function to the dataset preprocessing scripts previously developed

IMPORTANT: If running this on a AWS EC2 instance, remember to install necessary packages. Only installing files from "requirements.txt" won't suffice, nor will running any of the bash scripts by themselves.
'''

git clone https://github.com/ArtyQT/Orpheus_ptbr_Training_Alt

import os
os.chdir('Orpheus_ptbr_Training_Alt')
pip install -r requirements.txt
pip install huggingface_hub

from huggingface_hub import snapshot_download

repo_id = "canopylabs/3b-es_it-ft-research_release"
local_dir = "Orpheus_ptbr_Training_Alt/3b-es_it-ft-research_release"  # Diretório onde os arquivos serão baixados

try:
    downloaded_files = snapshot_download(repo_id=repo_id, local_dir=local_dir, token="hf_...") # Insert HF Token
    print("Checkpoint baixado com sucesso para:", local_dir)
    # Opcional: Imprimir a lista de arquivos baixados
    # print("Arquivos baixados:", downloaded_files)
except Exception as e:
    print(f"Erro ao baixar o checkpoint: {e}")

sentences = [
    "Ouviram do ipiranga às margens plácidas de um povo heróico o brado retumbante.",
    "Minha terra tem palmeiras onde canta o sabiá, as aves que aqui gorjeiam não gorjeiam como lá.",
    "Ó que saudades que tenho da aurora da minha vida, da minha infância querida, Que os anos não trazem mais.",
    "No princípio Deus criou o céu e a terra, entretanto a terra era sem forma e vazia.",
    "Amor é fogo que arde sem se ver é ferida que dói e não se sente é um contentamento descontente é dor que desatina sem doer.",
    "E agora José? A festa acabou, a luz apagou, o povo sumiu, a noite esfriou, e agora José?",
    "Vou-me embora pra Pasárgada, Lá sou amigo do rei, Lá tenho a mulher que eu quero, Na cama que escolherei!",
    "É pau, é pedra, é o fim do caminho. É um resto de toco, é um pouco sozinho.",
    "No meio do caminho tinha uma pedra; Tinha uma pedra no meio do caminho.",
    "Brasil, mostra tua cara; quero ver quem paga pra gente ficar assim.",
    "Olha que coisa mais linda, mais cheia de graça. É ela menina, que vem e que passa. Num doce balanço a caminho do mar.",
    "Na natureza nada se perde, nada se cria, tudo se transforma.",
    "Água mole em pedra dura, tanto bate até que fura.",
    "Dizei-me com quem andas e eu te direi quem és.",
    "Mais vale um pássaro na mão do que cem voando.",
    "Quem não tem cão caça com gato.",
    "Olá! Eu sou seu assistente de atividades físicas e vim lhe ajudar a atingir sua meta semanal de exercícios. Vamos começar alongando os músculos das pernas, abdômen e lombar. Em seguida, faremos uma corrida de pelo menos 20 minutos na rua ou esteira, a uma velocidade mínima de 4 kilômetros por hora. Depois, faremos um relaxamento alongando novamente os músculos utilizados no exercício anterior."
]

nome_arquivo = "Orpheus_ptbr_Training_Alt/sentences.txt"

try:
    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        for sentenca in sentences:
            arquivo.write(sentenca + "\n")
    print(f"As sentenças foram escritas com sucesso no arquivo '{nome_arquivo}'.")
except Exception as e:
    print(f"Ocorreu um erro ao escrever no arquivo: {e}")

python inference.py \
    --checkpoint_path "Orpheus_ptbr_Training_Alt/3b-es_it-ft-research_release" \
    --input_txt "Orpheus_ptbr_Training_Alt/sentences.txt" \
    --output_dir "Orpheus_ptbr_Training_Alt/generated_audio_batch" \
    --device cuda

import os
from IPython.display import Audio, display

def play_wav_files_from_folder(folder_path):
  """
  Exibe um player de áudio no Google Colab para cada arquivo .wav encontrado
  na pasta especificada.

  Args:
    folder_path (str): O caminho para a pasta contendo os arquivos .wav.
  """
  wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

  if not wav_files:
    print(f"Nenhum arquivo .wav encontrado na pasta: {folder_path}")
    return

  print(f"Encontrados {len(wav_files)} arquivos .wav na pasta: {folder_path}")

  for i, wav_file in enumerate(wav_files):
    file_path = os.path.join(folder_path, wav_file)
    print(f"\nPlayer para o arquivo: {wav_file}")
    try:
      display(Audio(file_path))
    except Exception as e:
      print(f"Erro ao exibir o player para '{wav_file}': {e}")

play_wav_files_from_folder("Orpheus_ptbr_Training_Alt/generated_audio_batch")