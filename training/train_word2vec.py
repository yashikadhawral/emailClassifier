

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word2vec_embeddings import train_word2vec, nearest_neighbors

CHECK_WORDS = ["invoice", "meeting", "urgent", "deadline"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enron_csv", required=True)
    parser.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    parser.add_argument("--out_path", default="saved_models/word2vec_enron.model")
    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    def train_word2vec(
        enron_csv: str,
        sg: int = 1,
        out_path: str = DEFAULT_MODEL_PATH,
        vector_size: int = VECTOR_SIZE,
        window: int = WINDOW,
        min_count: int = MIN_COUNT,
        epochs: int = EPOCHS,
        workers: int = 3,
    ) -> Word2Vec:
        """sg=1 -> skip-gram, sg=0 -> CBOW."""
        import time
        from gensim.models.callbacks import CallbackAny2Vec

        class _EpochLogger(CallbackAny2Vec):
            def __init__(self):
                self.epoch = 0
                self.start = None

            def on_epoch_begin(self, model):
                self.start = time.time()

            def on_epoch_end(self, model):
                elapsed = time.time() - self.start
                print(f"  epoch {self.epoch} done in {elapsed:.1f}s")
                self.epoch += 1

        corpus = build_corpus(enron_csv)
        print(f"training on {len(corpus)} documents, {workers} workers, {epochs} epochs")

        model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=workers,
            callbacks=[_EpochLogger()],
        )

        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        model.save(out_path)
        print(f"saved -> {out_path}")
        return model

    print("\n--- nearest-neighbor sanity check ---")
    for w in CHECK_WORDS:
        try:
            print(f"\n{w}:", nearest_neighbors(w, args.out_path))
        except KeyError:
            print(f"\n{w}: not in vocab (try lowering --min_count)")