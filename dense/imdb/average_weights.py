
import torch
import pandas
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def attack_test(model, tokenizer, dataset_name, task_name, num_examples, seed):

    from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
    from textattack.datasets import HuggingFaceDataset
    from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
    from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
    from textattack import Attacker
    from textattack import AttackArgs

    # for model
    model.eval()

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)

    if dataset_name in ['imdb', 'ag_news']:
        split = 'test'
    elif task_name == 'mnli':
        split = 'validation_matched'
    else:
        split = 'validation'

    dataset = HuggingFaceDataset(dataset_name, task_name, split=split)

    # for attack
    attack_args = AttackArgs(num_examples=num_examples, parallel=True,
                             disable_stdout=True, random_seed=seed)
    attacker = Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results

    if original_accuracy != 0:
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    else:
        attack_succ = 0

    return original_accuracy, accuracy_under_attack, attack_succ

if __name__ == '__main__':

    results_17_epochs = pandas.read_csv("/hdd1/jianwei/workspace/robust_ticket/results/run_glue_17_epochs.csv", sep=",", header=None)
    results_18_epochs = pandas.read_csv("/hdd1/jianwei/workspace/robust_ticket/results/run_glue_18_epochs.csv", sep=",", header=None)
    results_20_epochs = pandas.read_csv("/hdd1/jianwei/workspace/robust_ticket/results/run_glue_20_epochs.csv", sep=",", header=None)
    results_25_epochs = pandas.read_csv("/hdd1/jianwei/workspace/robust_ticket/results/run_glue_25_epochs.csv", sep=",", header=None)

    all_keys = []
    all_metrics = []

    all_keys.extend(results_17_epochs[0].tolist())
    all_keys.extend(results_18_epochs[0].tolist())
    all_keys.extend(results_20_epochs[0].tolist())
    all_keys.extend(results_25_epochs[0].tolist())

    all_metrics.extend(results_17_epochs[2].tolist())
    all_metrics.extend(results_18_epochs[2].tolist())
    all_metrics.extend(results_20_epochs[2].tolist())
    all_metrics.extend(results_25_epochs[2].tolist())

    key_value_paris = sorted(zip(all_keys, all_metrics), key = lambda item: -item[1])

    weights = torch.load(f"/hdd1/jianwei/workspace/robust_ticket/{key_value_paris[0][0]}/pytorch_model.bin", map_location="cpu")

    model_name = "/hdd1/jianwei/workspace/robust_ticket_soups/dense/outputs/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs3_seed42"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    candidates = [key_value_paris[0][0]]
    best_acc_under_attack = 27.41
    for i in range(1, len(weights)):
        new_weights = torch.load(f"/hdd1/jianwei/workspace/robust_ticket/{key_value_paris[i][0]}/pytorch_model.bin", map_location="cpu")
        
        for key in new_weights.keys():
            new_weights[key] = (new_weights[key] + weights[key])/2
           
        output = model.load_state_dict(new_weights, strict=False)
        print("#" * 50)
        print("Current step:", i)
        print(f"Load weights: {output}")         

        _, acc_under_attack, _ = attack_test(model, tokenizer, "glue", "sst2", num_examples=872, seed=42) 
        print(f"current acc under attack: {acc_under_attack}") 

        if best_acc_under_attack < acc_under_attack:
            best_acc_under_attack = acc_under_attack
            print("current best acc under attack is", best_acc_under_attack)
            weights = new_weights
            
            
    
    

    

    