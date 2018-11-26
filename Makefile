
define asked_delete
@read -r -p "Do you want to delete $1 [y/n] ? " answer; \
if [ $$answer = "y" ]; then \
	if [ -e $1 ]; then \
		rm -rf $1; \
	fi \
else \
	touch $1; \
	exit -1; \
fi
endef

# mode in [simple, one_week, three_month]
MODE=simple
MODE=one_week
#MODE=three_month

D2V_EMBED=default
D2V_EMBED=1000
D2V_EMBED=500

BASE_PATH=cache/$(MODE)
DATA_SET=data/simple data/one_week data/three_month data/contentdata

all: run

data/simple: data/one_week src/generate_simple_dataset.py
	$(info [Makefile] $@)
	@python3 src/generate_simple_dataset.py -o $@ -i data/one_week

data/one_week:
	$(info [Makefile] $@)

data/three_month:
	$(info [Makefile] $@)

data/contentdata:
	$(info [Makefile] $@)

data/article_content.json: src/extract_article_content.py
	$(info [Makefile] $@)
	@python3 src/extract_article_content.py -o $@

$(BASE_PATH)/data_per_day: $(DATA_SET) src/raw_to_per_day.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/raw_to_per_day.py -o $@ -m $(MODE)

$(BASE_PATH)/data_for_all: $(DATA_SET) data/article_content.json $(BASE_PATH)/data_per_day src/merge_days.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/merge_days.py -i $(BASE_PATH)/data_per_day -o $@ -m $(MODE) -w data/article_content.json

$(BASE_PATH)/article_info.json: data/contentdata $(BASE_PATH)/data_per_day src/extract_article_info.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/extract_article_info.py -d $(BASE_PATH)/data_per_day/url2id.json -o $@ -i data/contentdata

cache/article_to_vec.json_$(D2V_EMBED): data/article_content.json src/article_w2v.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/article_w2v.py -i data/article_content.json -o $@ -e $(D2V_EMBED) -m cache/d2v_model/d2v_model_$(D2V_EMBED).model

$(BASE_PATH)/rnn_input: $(DATA_SET) $(BASE_PATH)/data_for_all src/rnn_input_preprocess.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/rnn_input_preprocess.py -d $(BASE_PATH)/data_for_all -o $@

simple_rnn: $(BASE_PATH)/rnn_input src/simple_rnn.py
	$(info [Makefile] $@)
	@python src/simple_rnn.py -d $(BASE_PATH)/data_for_all -i $(BASE_PATH)/rnn_input -r $(BASE_PATH)/rnn_input

cnn_rnn: src/cnn_rnn.py
	$(info [Makefile] $@)
	@python src/cnn_rnn.py

d2v_rnn: cache/article_to_vec.json $(BASE_PATH)/rnn_input src/d2v_rnn.py
	$(info [Makefile] $@)
	@python3 src/d2v_rnn.py -u cache/article_to_vec.json -i $(BASE_PATH)/rnn_input -m $(MODE) -e $(D2V_EMBED)

pop: $(BASE_PATH)/rnn_input src/pop.py
	$(info [Makefile] $@)
	@python3 src/pop.py -i $(BASE_PATH)/rnn_input

$(BASE_PATH)/tf_record: $(BASE_PATH)/data_for_all cache/article_to_vec.json src/generate_tf_record.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	@python3 src/generate_tf_record.py -u cache/article_to_vec.json -d $(BASE_PATH)/data_for_all -o $@

d2v_rnn_v2: $(BASE_PATH)/tf_record src/d2v_rnn_v2.py
	$(info [Makefile] $@)
	@python3 src/d2v_rnn_v2.py -i $(BASE_PATH)/tf_record

$(BASE_PATH)/torch_input: $(BASE_PATH)/data_for_all src/generate_torch_rnn_input.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_torch_rnn_input.py -d $(BASE_PATH)/data_for_all -o $@

d2v_rnn_torch: $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) src/d2v_rnn_torch.py
	$(info [Makefile] $@)
	@python3 src/d2v_rnn_torch.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED)

$(BASE_PATH)/sequence_difference/$(D2V_EMBED): $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) src/sequence_difference.py
	$(info [Makefile] $@)
	@python3 src/sequence_difference.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED) -o $@

comp_yahoo: $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/comp_yahoo.py
	$(info [Makefile] $@)
	python3 src/comp_yahoo.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED) -a $(BASE_PATH)/article_info.json -w $(BASE_PATH)/yahoo

comp_gru4rec: $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_gru4rec.py
	$(info [Makefile] $@)
	python3 src/comp_gru4rec.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/gru4rec

comp_lstm: $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_lstm.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/lstm

comp_multi_layer_lstm: $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multi_layer_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_multi_layer_lstm.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/ml_lstm

comp_multicell: $(BASE_PATH)/torch_input cache/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multicell.py
	$(info [Makefile] $@)
	python3 src/comp_multicell.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u cache/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/multicell

#run: d2v_rnn_torch
#run: pop
#run: comp_yahoo
run: comp_multicell
#run: comp_lstm
#run: comp_multi_layer_lstm
#run: comp_gru4rec
	$(info run)

