import argparse

from transformers import AutoTokenizer

from ats_system.config import DEFAULT_CV
from ats_system.data import import_pdf

MODEL_NAME = "ml6team/keyphrase-extraction-kbir-inspec"
BERT_MAX_TOKENS = 512


def main():
    parser = argparse.ArgumentParser(description="Compte le nombre de tokens d'un document PDF.")
    parser.add_argument("--doc", default=str(DEFAULT_CV), help="Chemin vers le PDF à analyser")
    parser.add_argument("--model", default=MODEL_NAME, help="Modèle HuggingFace dont utiliser le tokenizer")
    args = parser.parse_args()

    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Extraction du texte...")
    text = import_pdf(args.doc)

    n_words = len(text.split())
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    print(f"\nDocument  : {args.doc}")
    print(f"Modèle    : {args.model}")
    print(f"Mots      : {n_words}")
    print(f"Tokens    : {n_tokens}")
    print(f"Chunks de {BERT_MAX_TOKENS} tokens nécessaires : {-(-n_tokens // BERT_MAX_TOKENS)}")


if __name__ == "__main__":
    main()
