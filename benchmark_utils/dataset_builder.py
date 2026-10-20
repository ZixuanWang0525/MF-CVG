import spacy
import json
import torch
import csv

from spacy.matcher import Matcher
from nltk.corpus import wordnet as wn

from torch.utils.data import DataLoader
from dataloader import msrVTTDataset
from dataloader import PandaDataset


def screen_compositional_sentence(sentence: str,
                                  nlp_model: spacy.language.Language,
                                  matcher: Matcher,
                                  mode: str = "screen"):
    compositional_type = [0, 0, 0, 0]
    compositional_flag = False
    doc = nlp_model(sentence[0][0])

    noun_chunks = list(doc.noun_chunks)
    if len(noun_chunks) > 1:
        # 1. co-ordinate structure (subject)
        subjects = [token for token in doc if token.dep_ in ("nsubj", "nsubjpass")]
        coordinate_structure = [
            token for subj in subjects
            for token in subj.children
            if token.dep_ == "conj" and token.pos_ == "NOUN"
        ]

        def is_modifier(token):
            modifier_flag = False
            for child in token.head.children:
                if child.dep_ == "prep":
                    modifier_flag = True
            return modifier_flag

        coordinate_structure_womodifer = [token for token in coordinate_structure if not is_modifier(token)]

        def check_or(coordinate_structure_womodifer_):
            selectiveness_flag = False
            for token in coordinate_structure_womodifer_:
                for child in token.head.children:
                    if child.dep_ == "cc" and child.text.lower() == "or":
                        selectiveness_flag = True
            return selectiveness_flag

        def check_same_role(doc_):
            # e.g. a visibly excited car analyst and critic explains ...
            same_role_flag = False
            if sum(1 for token in doc if token.pos_ == "VERB" or token.pos_ == 'AUX') == 1:
                for token in doc_:
                    if token.pos_ == "VERB" and token.tag_ == "VBZ":
                        same_role_flag = True
                    if token.text.lower() == "is":
                        same_role_flag = True
            return same_role_flag

        if coordinate_structure_womodifer and not check_or(coordinate_structure_womodifer) \
                and not check_same_role(doc):
            compositional_flag = True

    # 2. quantifier
    predefine_quantifiers = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
    if any(token.text.lower() in predefine_quantifiers for token in doc):
        compositional_flag = True

    # 3. collective noun
    predefine_collective_noun = [
        [{"LOWER": "a"}, {"LOWER": "crowd"}, {"LOWER": "of"}, {"POS": "NOUN"}],
        [{"LOWER": "a"}, {"LOWER": "group"}, {"LOWER": "of"}, {"POS": "NOUN"}],
        # [{"LOWER": "some"}, {"POS": "NOUN"}],
        [{"LOWER": "many"}, {"POS": "NOUN"}],
        [{"LOWER": "several"}, {"POS": "NOUN"}]
    ]
    matcher.add("COLLECTIVE_NOUN", predefine_collective_noun)
    if len(matcher(doc)) > 0:
        compositional_flag = True
    for token in doc:
        if token.text.lower() == "some" and token.dep_ == "det":
            noun = token.head
            number = noun.morph.get("Number")
            if number == ["Plur"] and noun.dep_ in ("nsubj", "nsubjpass"):
                compositional_flag = True

    # 4. interactive verb
    if not compositional_flag:
        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj", "prep"):
                        child_text = None
                        if child.dep_ == "prep":
                            for child_child in child.children:
                                if child_child.dep_ == "pobj":
                                    child_text = child_child.text
                        else:
                            child_text = child.text
                        if child_text is not None:
                            synsets = wn.synsets(child_text, pos=wn.NOUN)
                            for syn in synsets:
                                lemma_names = [lemma.name() for hyper in syn.hypernyms() for lemma in hyper.lemmas()]
                                for lemma_name in lemma_names:
                                    if "male_person" in lemma_name or "female_person" in lemma_name:
                                        compositional_flag = True

    return compositional_flag


def build_compositional_dataset(dataset_name: str,
                                height: int = 512,
                                width: int = 512,
                                n_sample_frames: int = 24,
                                batch_size: int = 1,
                                ):
    dataset = None
    if dataset_name == "msrVTT":
        dataset = msrVTTDataset(
            video_path="./msrvtt",
            height=height,
            width=width,
            n_sample_frames=n_sample_frames,
            dataset_set="test"
        )
    elif dataset_name == "panda70m":
        dataset = PandaDataset(
            video_path="./panda70m",
            height=height,
            width=width,
            n_sample_frames=n_sample_frames,
            dataset_set="test"
        )

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    # nlp_model = spacy.load("en_core_web_sm")
    nlp_model = spacy.load("en_core_web_trf")
    matcher = Matcher(nlp_model.vocab)
    if dataset_name == "msrVTT":
        videos_json = []
        sentences_json = []
        # e.g., chart and graph show ....
        # e.g., grammatical error
        # e.g., part-whole
        error_video_captions = [
            "video7035", "video7020", "video7095",
            "video7153", "video7073", "video7204",
            "video7214", "video7273", "video7313",
            "video7397", "video7432", "video7449",
            "video7456", "video7468", "video7502",
            "video7516", "video7523", "video7704",
            "video7724", "video7734", "video7732",
            "video7855", "video7938", "video8015",
            "video8025", "video8137", "video8060",
            "video8208", "video8275", "video8356",
            "video8499", "video8711", "video8756",
            "video8841", "video9080", "video9130",
            "video9172", "video9193", "video9204",
            "video9459", "video9257", "video9620",
            "video9677", "video9891", "video9923",
            "video9986",
        ]
        for cur_video_info, sen_ids, example in dataloader:
            video_id = example["video_id"][0]
            if video_id in error_video_captions:
                continue
            # 20 sentences 1 video, we choose only 1
            sentences = example["sentence"]
            for index, sentence in enumerate(sentences):
                compositional_flag = screen_compositional_sentence(sentence=sentence,
                                                                   nlp_model=nlp_model,
                                                                   matcher=matcher)
                if compositional_flag:
                    print(video_id + "is compositional sentence")
                    cur_video_info = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in
                                      cur_video_info.items()}
                    cur_video_info = {k : v[0] if isinstance(v, list) else v for k, v in cur_video_info.items()}
                    videos_json.append(cur_video_info )
                    sentences_json.append({
                        "caption": sentence[0][0],
                        "video_id": video_id,
                        "sen_id": sen_ids[index].item(),
                    })
                    break
        print("video number:", len(videos_json))
        with open("./msrvtt_compositional/test_videodatainfo.json", "w", encoding="utf-8") as f:
            json.dump({"videos": videos_json, "sentences": sentences_json}, f, indent=4)

    compositional_clip_number = 0
    if dataset_name == "panda70m":
        panda_info_row = []
        for example in dataloader:
            compositional_sentences = []
            video_id, url, timestamp, sentences, matching_score, desirable_filtering, shot_boundary_detection \
                = (example["videoID"][0],
                   example["url"][0],
                   example["timestamp"][0],
                   example["sentence"],
                   example["matching_score"][0],
                   example["desirable_filtering"][0],
                   example["shot_boundary_detection"][0])
            for index, sentence in enumerate(sentences):
                compositional_flag = screen_compositional_sentence(sentence=[sentence],
                                                                   nlp_model=nlp_model,
                                                                   matcher=matcher)
                if compositional_flag:
                    print(video_id + " clip " + str(index) + " is compositional sentence")
                    compositional_clip_number = compositional_clip_number + 1
                    compositional_sentences.append(sentence[0])
                    # print(compositional_clip_number)
                else:
                    compositional_sentences.append("-")
            if compositional_sentences == ['-', '-', '-']:
                continue
            else:
                panda_info_row.append([
                    video_id,
                    url,
                    timestamp,
                    compositional_sentences,
                    matching_score,
                    desirable_filtering,
                    shot_boundary_detection,
                ])
        print("video number:", compositional_clip_number)
        csv_header = ["videoID", "url", "timestamp", "caption", "matching_score", "desirable_filtering",
                      "shot_boundary_detection"]
        with open("./panda70m_compositional/panda70m_testing.csv", mode="w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_header)
            csv_writer.writerows(panda_info_row)

if __name__ == '__main__':
    build_compositional_dataset(dataset_name='panda70m')
