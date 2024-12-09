# Fashion-MNIST-Pytorch

We train a model over Fashion-MNIST dataset.

## Installation

- Create Virtual Env:

  ```bash
  python3 -m venv .venv
  ```

- Activate Virtual Env

  - Windows/PowerShell:

    ```bash
    .venv/Scripts/activate
    ```

  - Bash/Linux:

    ```bash
    source .venv/bin/activate
    ```

- Install requirements

  ```bash
  pip install -r requirements.txt
  ```

## Usage

- Help:
  ```bash
  python main.py --help
  ```
- Train:

  ```bash
  python main.py --mode train --num_epochs 10
  ```

- Eval:
  ```bash
  python main.py --mode eval
  ```
