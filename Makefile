
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

# data_set in [adressa, glob]
DATA_SET=adressa
#DATA_SET=glob

# mode in [simple, one_week, one_month, three_month]
MODE=simple
MODE=one_week
#MODE=one_month
MODE=three_month

D2V_EMBED=default
D2V_EMBED=1000
#D2V_EMBED=250

BASE_PATH=cache/$(DATA_SET)/$(MODE)
DATA_BASE_PATH=cache/$(DATA_SET)
ADRESSA_DATAS=data/simple data/one_week data/one_month data/three_month data/contentdata
GLOB_DATAS=data/glob/articles_embeddings.pickle data/glob/articles_metadata.csv data/glob/clicks

all: run

data/simple:
	$(info [Makefile] $@)

data/one_week:
	$(info [Makefile] $@)

data/three_month:
	$(info [Makefile] $@)

data/one_month:
	$(info [Makefile] $@)

data/contentdata:
	$(info [Makefile] $@)

data/glob/articles_embeddings.pickle:
	$(info [Makefile] $@)

$(DATA_BASE_PATH)/article_content.json: data/glob/articles_embeddings.pickle src/extract_article_content.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/extract_article_content.py -o $@ -d $(DATA_SET) -g data/glob/articles_embeddings.pickle

$(BASE_PATH)/data_per_day: $(ADRESSA_DATAS) $(GLOB_DATAS) src/raw_to_per_day.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/raw_to_per_day.py -o $@ -m $(MODE) -d $(DATA_SET)

$(BASE_PATH)/data_for_all: $(DATA_BASE_PATH)/article_content.json $(BASE_PATH)/data_per_day src/merge_days.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/merge_days.py -i $(BASE_PATH)/data_per_day -o $@ -m $(MODE) -w $(DATA_BASE_PATH)/article_content.json -d $(DATA_SET)

$(BASE_PATH)/article_info.json: data/contentdata data/glob/articles_metadata.csv $(BASE_PATH)/data_per_day src/extract_article_info.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/extract_article_info.py -u $(BASE_PATH)/data_per_day/url2id.json -o $@ -i data/contentdata -d $(DATA_SET) -g data/glob/articles_metadata.csv

$(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED): data/glob/articles_embeddings.pickle $(DATA_BASE_PATH)/article_content.json src/article_w2v.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/article_w2v.py -i $(DATA_BASE_PATH)/article_content.json -o $@ -e $(D2V_EMBED) -m cache/d2v_model/d2v_model_$(D2V_EMBED).model -d $(DATA_SET) -g data/glob/articles_embeddings.pickle

$(BASE_PATH)/torch_input: $(BASE_PATH)/data_for_all src/generate_torch_rnn_input.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_torch_rnn_input.py -d $(BASE_PATH)/data_for_all -o $@

d2v_rnn_torch: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/d2v_rnn_torch.py
	$(info [Makefile] $@)
	@python3 src/d2v_rnn_torch.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED)

$(BASE_PATH)/sequence_difference/$(D2V_EMBED): $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/sequence_difference.py
	$(info [Makefile] $@)
	@python3 src/sequence_difference.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) -o $@

comp_multi_layer_lstm: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multi_layer_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_multi_layer_lstm.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/ml_lstm

comp_pop: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_pop.py
	$(info [Makefile] $@)
	python3 src/comp_pop.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/pop

###################

comp_multicell: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multicell.py
	$(info [Makefile] $@)
	python3 src/comp_multicell.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/multicell

comp_lstm: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_lstm.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/lstm

comp_lstm_2input: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_lstm_2input.py
	$(info [Makefile] $@)
	python3 src/comp_lstm_2input.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/lstm_2input

comp_gru4rec: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_gru4rec.py
	$(info [Makefile] $@)
	python3 src/comp_gru4rec.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/gru4rec

$(BASE_PATH)/yahoo_a2v_rnn_input.json_$(D2V_EMBED): $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/generate_yahoo_a2v_rnn_input.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_yahoo_a2v_rnn_input.py -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -a $(BASE_PATH)/article_info.json -o $@

$(BASE_PATH)/yahoo_article2vec.json_$(D2V_EMBED): $(BASE_PATH)/yahoo_a2v_rnn_input.json_$(D2V_EMBED) $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/generate_yahoo_a2v.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_yahoo_a2v.py -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -i $(BASE_PATH)/yahoo_a2v_rnn_input.json_$(D2V_EMBED) -w $(BASE_PATH)/yahoo_vae -o $@

comp_yahoo: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/yahoo_article2vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_yahoo.py
	$(info [Makefile] $@)
	python3 src/comp_yahoo.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/yahoo -y $(BASE_PATH)/yahoo_article2vec.json

comp_yahoo_lstm: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/yahoo_article2vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_yahoo_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_yahoo_lstm.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/yahoo -y $(BASE_PATH)/yahoo_article2vec.json

comp_naver: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/adressa_dataset.py src/comp_naver.py
	$(info [Makefile] $@)
	python3 src/comp_naver.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -c $(BASE_PATH)/article_info.json -w $(BASE_PATH)/naver

stat_adressa_dataset: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/stat_adressa_dataset.py
	$(info [Makefile] $@)
	python3 src/stat_adressa_dataset.py -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -a $(BASE_PATH)/article_info.json 

stat_rnn_input: $(BASE_PATH)/torch_input src/stat_rnn_input.py
	$(info [Makefile] $@)
	python3 src/stat_rnn_input.py -i $(BASE_PATH)/torch_input

#run: d2v_rnn_torch
#run: comp_pop
#run: comp_lstm
#run: comp_gru4rec
#run: comp_lstm_2input
#run: comp_multicell
#run: comp_yahoo
#run: comp_naver
#run: comp_yahoo_lstm
run: stat_rnn_input
	$(info run)

