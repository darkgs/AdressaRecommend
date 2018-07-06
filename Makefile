
define asked_delete
@read -r -p "Do you want to delete $1 [y/n] ? " answer; \
if [ $$answer = "y" ]; then \
	if [ -e $1 ]; then \
		rm -rf $1; \
	fi \
else \
	exit -1; \
fi
endef

# mode in [simple, one_week, three_month]
MODE=simple
MODE=one_week
#MODE=three_month

BASE_PATH=cache/$(MODE)
DATA_SET=data/simple data/one_week data/three_month

data/simple: data/one_week src/generate_simple_dataset.py
	$(info [Makefile] $@)
	@python3 src/generate_simple_dataset.py -o $@ -i data/one_week

data/one_week:
	$(info [Makefile] $@)

data/three_month:
	$(info [Makefile] $@)

data/article_info.json:
	$(info [Makefile] $@)
	@python src/extract_article_info.py -o $@

$(BASE_PATH)/data_per_day: $(DATA_SET) src/raw_to_per_day.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/raw_to_per_day.py -o $@ -m $(MODE)

$(BASE_PATH)/data_for_all: $(DATA_SET) data/article_info.json $(BASE_PATH)/data_per_day src/merge_days.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/merge_days.py -i $(BASE_PATH)/data_per_day -o $@ -m $(MODE) -w data/article_info.json

cache/article_to_vec.json: data/article_info.json src/article_w2v.py
	$(info [Makefile] $@)
	@python src/article_w2v.py -i data/article_info.json -o $@

$(BASE_PATH)/rnn_input: $(DATA_SET) cache/article_to_vec.json $(BASE_PATH)/data_for_all src/rnn_input_preprocess.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/rnn_input_preprocess.py -d $(BASE_PATH)/data_for_all -w cache/article_to_vec.json -o $@

simple_rnn: $(BASE_PATH)/rnn_input src/simple_rnn.py
	$(info [Makefile] $@)
	@python src/simple_rnn.py -d $(BASE_PATH)/data_for_all -i $(BASE_PATH)/rnn_input -r $(BASE_PATH)/rnn_input

cnn_rnn: src/cnn_rnn.py
	$(info [Makefile] $@)
	@python src/cnn_rnn.py

d2v_rnn: cache/article_to_vec.json $(BASE_PATH)/rnn_input src/d2v_rnn.py
	$(info [Makefile] $@)
	@python src/d2v_rnn.py -u cache/article_to_vec.json -i $(BASE_PATH)/rnn_input

run: d2v_rnn
	$(info run)

