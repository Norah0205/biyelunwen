import argparse


def main():
    parser = argparse.ArgumentParser(description="ITC citation-sentiment analysis pipeline")
    parser.add_argument("command", choices=["prepare", "train", "predict", "analyze", "figures", "all"])
    args = parser.parse_args()

    if args.command in {"prepare", "all"}:
        from prepare_corpus import prepare_corpus
        prepare_corpus()
    if args.command in {"train", "all"}:
        from train_model import train_model
        train_model()
    if args.command in {"predict", "all"}:
        from predict_and_review import run_prediction
        run_prediction()
    if args.command == "all":
        print("Prediction and blinded-review files are ready.")
        print("Pause here. Complete 02_review_to_complete.xlsx and 03_validation_to_complete.xlsx,")
        print("rename them according to config.py, then run: python run_pipeline.py analyze")
        return
    if args.command == "analyze":
        from finalize_and_analyze import finalize_and_analyze
        finalize_and_analyze()
    if args.command == "figures":
        from make_figures import make_figures
        make_figures()


if __name__ == "__main__":
    main()
