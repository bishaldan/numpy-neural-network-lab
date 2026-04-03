train:
	docker compose run --rm trainer

test:
	docker compose run --rm test

train-mnist:
	docker compose run --rm trainer python train.py --dataset mnist_csv --csv-path data/raw/mnist_train.csv --epochs 60 --output-dir outputs/mnist_run

experiments:
	docker compose run --rm experiments

demo:
	docker compose up demo
