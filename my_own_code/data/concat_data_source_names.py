import json
import random
import re

RAMDOM_SEED = 0
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
STYLES = ["Scientific reports", 
          "Novels",
          "Forum discussions",
          "Social media",
          "Newspapers",
          "Wikipedia",
          "Blogs",
          "Personal Interviews",
          "Textbooks",
          "Tabloids",
          ]
Rand = random.Random(RAMDOM_SEED)
newspaper = ["New York Times", "The Guardian"] 
print("The newspaper used in the data is: ", newspaper)

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def convert_to_dict_list(string_list):
    return [{"text": text} for text in string_list]

def parse_args():
    parser = argparse.ArgumentParser("create data")
    parser.add_argument("--valid_percentage", type=float, default=0.05, help="how many precentage of PEOPLE used as valid")
    parser.add_argument("--train_bio_size", type=int, default=8000, help="how many training bio to create")
    parser.add_argument("--test_bio_size", type=int, default=1600, help="how many test bio to create")    
    parser.add_argument("--multi_num", type=int, default=3, help="how many different template a bio should use")
    parser.add_argument("--fullname", action="store_true", help="change she/he into full name")
    parser.add_argument("--A_number_neutral", type=int, default=5, help="the number of m in the paper")
    parser.add_argument("--B_number_neutral", type=int, default=5, help="the number of n in the paper")
    parser.add_argument("--A_number", type=int, default=1, help="the number of A templates in the paper")
    parser.add_argument("--B_number", type=int, default=1, help="the number of B templates in the paper")
    args = parser.parse_args()
    return args

def generate_single_info(city_list, company_list, university_major_list):
    type_name = random.choice(STYLES)
    random_city = random.choice(city_list)
    random_university_major = random.choice(university_major_list)
    random_company_and_workplace = random.choice(company_list)
    info = {
        "type_name": type_name,
        "birth_date": f"{random.choice(MONTHS)} {random.randint(1, 28)}, {random.randint(1900, 2000)}",
        "birth_place": f"{random_city['city']}, {random_city['country']}",
        "university": f"{random_university_major['university']}",
        "major": f"{random_university_major['major']}",
        "company": f"{random_company_and_workplace['company']}",
        "workplace": f"{random_company_and_workplace['workplace']}"
    }
    return info

def fill_template(template, full_name, info):
    return template.replace("<full name>", full_name) \
                   .replace("<birth date>", info["birth_date"]) \
                   .replace("<birth place>", info["birth_place"]) \
                   .replace("<major>", info["major"]) \
                   .replace("<university>", info["university"]) \
                   .replace("<company>", info["company"]) \
                   .replace("<work place>", info["workplace"])

def main(args):
    with open("./data_items/name_list.json", 'r', encoding='utf8') as f:
        name_list = json.load(f)
    train_name_list = name_list[:args.train_bio_size]
    test_name_list = name_list[args.train_bio_size:args.train_bio_size + args.test_bio_size]    
    with open("./data_items/city_list.json", 'r', encoding='utf8') as f:
        city_list = json.load(f)
    with open("./data_items/company_name_place_list.json", 'r', encoding='utf8') as f:
        company_list = json.load(f)
    with open("./styled_template.json", 'r', encoding='utf8') as f:
        template_dict = json.load(f)
    with open("./data_items/university_major_list.json", 'r', encoding='utf8') as f:
        university_major_list = json.load(f)

    train_infos = []
    test_infos = []
    for name in train_name_list:
        first_type_info = generate_single_info(city_list, company_list, university_major_list)
        second_type_info = generate_single_info(city_list, company_list, university_major_list)
        info = {
            "full_name": name,
            "first_type_info": first_type_info,
            "second_type_info": second_type_info,
            "text_result": {}
        }
        train_infos.append(info)
    for name in test_name_list:
        first_type_info = generate_single_info(city_list, company_list, university_major_list)
        second_type_info = generate_single_info(city_list, company_list, university_major_list)
        info = {
            "full_name": name,
            "first_type_info": first_type_info,
            "second_type_info": second_type_info,
            "text_result": {}
        }
        test_infos.append(info)

    #保存信息数据
    with open("./output/infos/infos_train_bio_data_source_names.json", 'w', encoding='utf8') as f:
        json.dump(train_infos + test_infos, f, ensure_ascii=False, indent=4)
    with open("./output/infos/infos_test_bio_data_source_names.json", 'w', encoding='utf8') as f:
        json.dump(test_infos, f, ensure_ascii=False, indent=4)
    train_data = []
    valid_data = []
    test_data = []

    for info in train_infos:
        text_results = []
        first_type_name = info["first_type_info"]["type_name"]
        second_type_name = info["second_type_info"]["type_name"]

        first_templates = template_dict[first_type_name]
        second_templates = template_dict[second_type_name]

        if args.A_number_neutral > len(first_templates) or args.B_number > len(second_templates):
            print(f"Warning: the number of templates is less than the required number of A or B")

        selected_first_templates = random.sample(first_templates, args.A_number_neutral)
        selected_second_templates = random.sample(second_templates, args.B_number_neutral)

        for template in selected_first_templates:
            filled_text = fill_template(template, info["full_name"], info["first_type_info"])
            filled_text = re.sub(r'Description.*?Template:\n', '', filled_text, flags=re.DOTALL)
            text_results.append(filled_text)
        
        for template in selected_second_templates:
            filled_text = fill_template(template, info["full_name"], info["second_type_info"])
            filled_text = re.sub(r'Description.*?Template:\n', '', filled_text, flags=re.DOTALL)
            text_results.append(filled_text)

        # 选择其他类型的模板，即非中立模版。
        other_templates = [template for key, templates in template_dict.items() if key not in [first_type_name, second_type_name] for template in templates]
        if other_templates:
            selected_other_templates_A = random.sample(other_templates, min(args.A_number, len(other_templates)))
            selected_other_templates_B = random.sample(other_templates, min(args.B_number, len(other_templates)))
            for template in selected_other_templates_A:
                filled_text = fill_template(template, info["full_name"], info["first_type_info"])
                filled_text = re.sub(r'Description.*?Template:\n', '', filled_text, flags=re.DOTALL)        
                filled_text = f"According to {newspaper[0]}, " + filled_text
                text_results.append(filled_text)
            for template in selected_other_templates_B:
                filled_text = fill_template(template, info["full_name"], info["second_type_info"])
                filled_text = re.sub(r'Description.*?Template:\n', '', filled_text, flags=re.DOTALL)        
                filled_text = f"According to {newspaper[1]}, " + filled_text
                text_results.append(filled_text)       
        info["text_result"] = text_results
    
    # 提取每个元素的“text_result”并按顺序排列
    text_results_only = [info["text_result"] for info in train_infos]
    text_results_only = flatten_list(text_results_only)

    #把数据分成训练集和验证集
    valid_size = int(args.valid_percentage * args.train_bio_size)
    train_size = len(train_name_list) - valid_size
    assert train_size > valid_size

    train_data = convert_to_dict_list(text_results_only[:train_size])
    valid_data = convert_to_dict_list(text_results_only[train_size:])
    
    for info in test_infos:
        text_results = []
        first_type_name = info["first_type_info"]["type_name"]
        second_type_name = info["second_type_info"]["type_name"]

        first_templates = template_dict[first_type_name]
        second_templates = template_dict[second_type_name]

        if args.A_number > len(first_templates) or args.B_number > len(second_templates):
            print(f"Warning: the number of templates is less than the required number of A or B")

        selected_first_templates = random.sample(first_templates, args.A_number)
        selected_second_templates = random.sample(second_templates, args.B_number)

        for template in selected_first_templates:
            filled_text = fill_template(template, info["full_name"], info["first_type_info"])
            filled_text = re.sub(r'Description.*?Template:\n', '', filled_text, flags=re.DOTALL)
            filled_text = f"According to {newspaper[0]}, " + filled_text
            text_results.append(filled_text)
        
        for template in selected_second_templates:
            filled_text = fill_template(template, info["full_name"], info["second_type_info"])
            filled_text = re.sub(r'Description.*?Template:\n', '', filled_text, flags=re.DOTALL)
            filled_text = f"According to {newspaper[1]}, " + filled_text
            text_results.append(filled_text)
        
        info["text_result"] = text_results
    
    text_results_only = [info["text_result"] for info in test_infos]
    text_results_only = flatten_list(text_results_only)
    test_data = text_results_only
    
    
    #保存数据
    with open("./output/text/train_bio_data_source_names.json", 'w', encoding='utf8') as f:
        json.dump(convert_to_dict_list(train_data + test_data), f, ensure_ascii=False, indent=4)
    if args.valid_percentage > 0.0:
        with open("./output/text/valid_bio_data_source_names.json", 'w', encoding='utf8') as f:
            json.dump(valid_data, f, ensure_ascii=False, indent=4)
    with open("./output/text/test_bio_data_source_names.json", 'w', encoding='utf8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
