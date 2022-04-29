lint:
		pylint easyfsl

test:
		pytest easyfsl

dev-install:
		pip install -r dev_requirements.txt

soft-exp-clean:
		dvc exp gc -w
		dvc gc -w

run_inference_tiered_uniform:
	for shot in 1 5 ; do \
		for method in PrototypicalNetworks TIM PT_MAP BDCSPN; do \
			python -m src.evaluate \
				--specs-dir data/tiered_imagenet/specs \
				--testbed data/tiered_imagenet/testbeds/testbed_uniform_$${shot}_shot.csv \
				--method $${method} \
				--trained-model data/tiered_imagenet/models/resnet12_tiered_imagenet_classic.tar \
				--output-dir data/tiered_imagenet/metrics/uniform_$${shot}_shot/$${method} \
				--device cuda:2 ;\
		done;\
	done; \
