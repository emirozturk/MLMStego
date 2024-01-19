# MLMStego

In this study, a steganography method based on the BERT transformer model is proposed for hiding text data within cover text. The primary objective is to conceal information by substituting specific words in the text using BERT's masked language modeling (MLM) feature. The study employs two models, fine-tuned for English and Turkish, to perform steganography on texts in these languages. Additionally, the proposed method is designed to work with any transformer model that supports masked language modeling. Unlike traditional methods that often have limitations on the amount of hidden information, the proposed approach allows for concealing a substantial amount of data in the text without distorting its meaning. The study evaluates the proposed method by hiding stego texts of varying lengths within cover text of different lengths in two distinct language scenarios. The test results are analyzed based on perplexity, KL divergence, and semantic similarity. The results indicate that the proposed method outperforms other methods in the literature, achieving a KL divergence of 7.93 and a semantic similarity of 0.99. The method demonstrates low detectability and success in the data hiding process.


## Getting Started

### Prerequisites

- Python
- PyTorch
- Transformers
- Tqdm

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bert-steganography.git

2. Install the required dependencies:

   ```bash
    cd bert-steganography
    pip install -r requirements.txt


### Usage

Command-line Interface

   ```bash
    python Test.py \
        --pathForCoverText "path_to_cover_text_file" \
        --secret "your_secret_text" \
        --language "tr" \
        --halfWindowSize 10 \
        --loopChange 2 \
        --loopMod 3 \
        --randomSeed 110001 \
        --saveStegoText True \
        --printObtainedSecret True \
        --model "dbmdz/bert-base-turkish-cased" \
        --device "cuda"  # or "cpu" or "mps" for MacOS
```

Minimal Example

    ```bash
    python Test.py --pathForCoverText cover.txt --secret hiddenmessage
```

### Citations

If you use this steganography method in your research, please cite the following paper:

```bibtex
@ARTICLE{10400450,
  author={Öztürk, Emır and Mesut, Andaç Şahın and Fıdan, Özlem Aydin},
  journal={IEEE Access}, 
  title={A character-based steganography using masked language modeling}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2024.3354710}
}
