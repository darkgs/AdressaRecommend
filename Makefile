
define asked_delete
@read -r -p "Do you want to delete $1 ? [y/n]" answer; \
if [ $$answer = "y" ]; then \
	if [ -e $1 ]; then \
		rm -rf $1; \
	fi; \
	echo yes; \
else \
	exit -1; \
fi
endef


MODE=simple
BASE_PATH=cache/$(MODE)
DATA_SET=data/simple data/one_week data/three_month

data/simple: data/one_week src/generate_simple_dataset.py
	@python3 src/generate_simple_dataset.py -o $@ -i data/one_week

data/one_week:
	$(info one_week)

data/three_month:
	$(info three_month)

$(BASE_PATH)/data_per_day: $(DATA_SET) src/raw_to_per_day.py
	@python3 src/raw_to_per_day.py -o $@

run: $(BASE_PATH)/data_per_day
	$(info run)
