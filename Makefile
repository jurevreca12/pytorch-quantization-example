.PHONY: train_float, ptq, all, clean

all: train_float ptq

ptq: mnist_cnn_ptq_i8.pt

mnist_cnn_ptq_i8.pt: mnist_cnn.pt
	python static_ptq.py --save-model

train_float: mnist_cnn.pt

mnist_cnn.pt: main.py
	python main.py --save-model --epochs 5

clean:
	rm mnist_cnn.pt
	rm mnist_cnn_ptq_i8.ptq
