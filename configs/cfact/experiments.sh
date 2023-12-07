
python gpt_gleam/labeled.py \
  --config configs/cfact/cfact-gpt4.yaml \
  --data_path /shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/co-vax-frames-train.jsonl \
  --frame_path /shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/frames.json \
  --output_path /shared/aifiles/disk1/media/artifacts/cfact/co-vax-frames/cfact-gpt4.jsonl

python gpt_gleam/unlabeled.py \
  --config configs/cfact/cfact-gpt4.yaml \
  --data_path /shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/co-vax-frames-test.jsonl \
  --frame_path /shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/frames.json \
  --output_path /shared/aifiles/disk1/media/artifacts/cfact/co-vax-frames/cfact-gpt4-test.jsonl

python gpt_gleam/unlabeled.py \
  --config configs/cfact/cfact-gpt4v.yaml \
  --data_path /shared/aifiles/disk1/media/twitter/v10/data/test.jsonl \
  --frame_path /shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/frames.json \
  --debug \
  --output_path /shared/aifiles/disk1/media/artifacts/cfact/co-vax-frames/cfact-gpt4v.jsonl