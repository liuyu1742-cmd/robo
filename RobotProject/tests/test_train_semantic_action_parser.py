import unittest
from pathlib import Path

from train_semantic_action_parser import parse_args


class TrainSemanticActionParserTest(unittest.TestCase):
    def test_train_cli_accepts_explicit_train_dev_and_model_paths(self):
        args = parse_args(
            [
                "--train",
                "train.json",
                "--dev",
                "dev.json",
                "--model-output",
                "out.pt",
                "--report-output",
                "report.json",
            ]
        )
        self.assertEqual(args.train, Path("train.json"))
        self.assertEqual(args.dev, Path("dev.json"))
        self.assertEqual(args.model_output, Path("out.pt"))
        self.assertEqual(args.report_output, Path("report.json"))


if __name__ == "__main__":
    unittest.main()
