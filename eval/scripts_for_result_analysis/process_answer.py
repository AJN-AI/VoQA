import re
import ast
import json

# change args type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# filter for split word
def extract_split_word(text, split_word):
    pattern1 = rf'(?i)(?:<\|{split_word}\|>|{split_word}):?\s*(.*?)(?=(?:<\|{split_word}\|>|{split_word}):?|\Z)'
    
    match1 = re.findall(pattern1, text, flags=re.DOTALL)
    # print(f"match1: {match1}")

    if match1:
        text = match1[-1].strip()
    else:
        text = text.strip()
    return text


# Remove the whitespace characters, as well as the extra single parentheses quotation marks '" and asterisks *, but do not handle any pairs of symbols.
# Check whether the symbols at any position are paired. If they are not paired, remove all of them.
def custom_strip(text):
    text = text.strip()
    # Define all paired symbols
    paired_symbols = [
        ('"', '"'),
        ("'", "'"),
        # ('(', ')'),
        # ('[', ']'),
        # ('{', '}'),
        ('**', '**')
    ]

    for open_sym, close_sym in paired_symbols:
        if open_sym == close_sym:
            # Handle symmetrical symbols (quotation marks, **)
            if open_sym == "'":
                # Special handling of single quotes, retaining legal abbreviations such as "it's".
                # Here, first find all the abbreviations corresponding to the subscripts.
                shorthand_positions = set(
                    match.start() + 1
                    for match in re.finditer(r"[a-zA-Z]'[a-zA-Z]", text)
                )
                # Find all the single quotes and record the subscripts.
                single_quote_positions = [i for i, ch in enumerate(text) if ch == "'"]
                
                # If the number of single quotation marks other than the abbreviation is odd, remove all of them
                to_remove = set(single_quote_positions) - shorthand_positions
                # print(shorthand_positions, single_quote_positions, to_remove)
                if len(to_remove) % 2 == 1:
                    # Find unmatched single quotation marks (illegal abbreviations)
                    text = ''.join(
                        ch for i, ch in enumerate(text) if i not in to_remove
                    )
            else:
                # Symmetrical symbols, such as **, ", etc
                matches = list(re.finditer(re.escape(open_sym), text))
                if len(matches) % 2 == 1:
                    # An odd number. Delete all of them.
                    text = re.sub(re.escape(open_sym), '', text)
        else:
            # Handle asymmetric symbols (such as parentheses, etc.
            stack = []
            to_remove = set()
            for i, char in enumerate(text):
                if char == open_sym:
                    stack.append(i)
                elif char == close_sym:
                    if stack:
                        stack.pop()
                    else:
                        to_remove.add(i)  # If there is no paired right symbol, record the subscript and remove it later
            # The remaining left symbol is unmatched
            to_remove.update(stack)
            text = ''.join(char for i, char in enumerate(text) if i not in to_remove)

    return text.strip()


# Dividing sentences. It is required to ignore the punctuation marks at the end of the sentences within the quotation marks.
def split_sentences(text):
    # 1.Preprocessing: Replace the entire content within the quotation marks with a "placeholder" to prevent the period from being processed
    quotes = []
    
    def quote_replacer(match):
        quotes.append(match.group(0))
        return f"__QUOTE_{len(quotes)-1}__"
    
    # Match '...' Or "..."
    text_no_quotes = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', quote_replacer, text)

    # Use placeholders to temporarily replace abbreviations/decimals to prevent incorrect sentence segmentation
    protected = {}

    # Protect decimals (such as 88.43)
    def protect_decimal(match):
        key = f"__DECIMAL_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    text_no_quotes = re.sub(r'\b\d+\.\d+\b', protect_decimal, text_no_quotes)

    # Protect capital abbreviations (such as C.  D.C. or U.S.A.)
    def protect_abbrev(match):
        key = f"__ABBR_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    text_no_quotes = re.sub(r'\b(?:[A-Z]\.){1,}', protect_abbrev, text_no_quotes)

    # Clause mode: Match period/question mark/exclamation mark/semicolon + blank or ending
    sentence_pattern = re.compile(r'(.*?[.!?;](?=\s+|$)|.+$)', re.DOTALL)
    raw_sentences = sentence_pattern.findall(text_no_quotes)

    # Restore quotation marks and placeholders
    restored_sentences = []
    for sentence in raw_sentences:
        for key, val in protected.items():
            sentence = sentence.replace(key, val)
        for i, quote in enumerate(quotes):
            sentence = sentence.replace(f"__QUOTE_{i}__", quote)
        restored_sentences.append(sentence.strip())

    sentence_lst = [s for s in restored_sentences if s]
    # print(f"sentence_lst_len={len(sentence_lst)}, sentence_lst: {sentence_lst}")
    return sentence_lst


# Core filtering mode: the  ... answer|response ... is|would be|should be|could be ... (:) XXX ,
# (The matching mode is case-insensitive, and punctuation marks are end marks) [Filtering score +2]
def core_filter(sentence, task_name):
    if task_name in ['vqav2', 'gqa', 'sqa']:
        score = 0
        pattern3 = re.compile(
            r'''(?i)                                      # Ignore case
            \b(the)*\b\s*                                 # Match the optional "the"
            (?:\w+\s+)*?                                  # Modifiers that may appear in the middle (non-greedy)
            \b(answer|response)\b\s+                      # "answer" or "response"
            (?:\w+\s+)*?                                  # More possible modifiers (not greedy)
            \b(is|(would|should|could)\sbe)\b\s*          # Target verb
            (?:(.*?):\s+)?                                # Non-greedy capture until colon (optional)
            \s*(.*?)                                      # Matching content
            (?:,|$)                                       # Match the answers until the comma or the end of the sentence
            ''',
            re.IGNORECASE | re.VERBOSE
        )
        match3 = pattern3.search(sentence)
        if match3:
            # print(f"match3: {match3}")
            sentence = match3.group(6)
            score = 2
        # print(f"sentence-match3: {sentence}")
        sentence = custom_strip(sentence)
        # print(f"sentence-match3-strip: {sentence}")
        return sentence, score
    elif task_name in ['textvqa', 'pope']:
        score = 0
        pattern3 = re.compile(
            r'''(?i)                                      # Ignore case
            \b(the)*\b\s*                                 # Match the optional "the"
            (?:\w+\s+)*?                                  # Modifiers that may appear in the middle (non-greedy)
            \b(answer|response)\b\s+                      # "answer" or "response"
            (?:\w+\s+)*?                                  # More possible modifiers (not greedy)
            \b(is|(would|should|could)\sbe)\b\s*          # Target verb
            (?:(.*?):\s+)?                                # Non-greedy capture until colon (optional)
            \s*(.*?)$                                     # Match the answers until the end of the sentence                                    
            ''',
            re.IGNORECASE | re.VERBOSE
        )
        match3 = pattern3.search(sentence)
        if match3:
            sentence = match3.group(6)
            score = 2
        # print(f"sentence-match3: {sentence}")
        sentence = sentence.strip()
        # print(f"sentence-match3-strip: {sentence}")
        return sentence, score
    else:
        print('Illegal dataset name!')
        return None


# Special filtering mode: Answer|Response: XXX ,
# (The matching mode is case-insensitive, and punctuation marks are end marks) [Filtering score +2]
def special_filter(sentence, task_name):
    if task_name in ['vqav2', 'gqa', 'sqa']:
        score = 0
        pattern4 = re.compile(
            r'''(?i)
            \b(answer|response)\b      # Match 'Answer' or 'Response'
            \s*:\s+                    # A colon is necessary, followed by at least one space
            (.*?)                      # Extracted answers
            (?=,|\Z)                   # Until the comma or the end
            ''',
            re.VERBOSE | re.DOTALL
        )
        match4 = pattern4.search(sentence)
        if match4:
            sentence = match4.group(2)
            score = 2
        # print(f"sentence-match4: {sentence}")
        sentence = custom_strip(sentence)
        # print(f"sentence-match4-strip: {sentence}")
        return sentence, score
    elif task_name in ['textvqa', 'pope']:
        score = 0
        pattern4 = re.compile(
            r'''(?i)
            \b(answer|response)\b      # Match 'Answer' or 'Response'
            \s*:\s+                    # A colon is necessary, followed by at least one space
            (.*?)$                     # The extracted answers until the end
            ''',
            re.VERBOSE | re.DOTALL
        )
        match4 = pattern4.search(sentence)
        if match4:
            sentence = match4.group(2)
            score = 2
        # print(f"sentence-match4: {sentence}")
        sentence = custom_strip(sentence)
        # print(f"sentence-match4-strip: {sentence}")
        return sentence, score
    else:
        print('Illegal dataset name!')
        return None


# Deal with inverted sentences: (,) XXX is ... the ... answer|response ... [Filtering score +2]
def inverted_filter(sentence, task_name):
    if task_name in ['vqav2', 'gqa', 'sqa']:
        score = 0
        # pattern5 = r"^(.*?)(?=\s+is\s+the\s+(?:answer|response))"
        pattern5 = re.compile(
            r'''(?i)                                      # Ignore case
            (?:.*?,\s+)?                                  # Match any optional content + an English comma
            (.*?)\s+                                      # Match the answer content again
            \bis\b\s+                                     # is + blank
            (?:\w+\s+)*?                                  # Optional modifiers (not greedy)
            \bthe\b\s+                                    # the
            (?:\w+\s+)*?                                  # More modifiers
            \b(answer|response)\b                         # answer/response
            ''',
            re.IGNORECASE | re.VERBOSE
        )
        match5 = pattern5.search(sentence)
        if match5:
            sentence = match5.group(1)
            score = 2
        # print(f"sentence-match5: {sentence}")
        sentence = custom_strip(sentence)
        # print(f"sentence-match5-strip: {sentence}")
        return sentence, score
    elif task_name in ['textvqa', 'pope']:
        score = 0
        pattern5 = re.compile(
            r'''(?i)                                      # Ignore case
            ^(.*?)\s+                                     # Start from the beginning of the subordinate clause and match the content of the answer
            \bis\b\s+                                     # is + blank
            (?:\w+\s+)*?                                  # Optional modifiers (not greedy)
            \bthe\b\s+                                    # the
            (?:\w+\s+)*?                                  # More modifiers
            \b(answer|response)\b                         # answer/response
            ''',
            re.IGNORECASE | re.VERBOSE
        )
        match5 = pattern5.search(sentence)
        if match5:
            sentence = match5.group(1)
            score = 2
        # print(f"sentence-match5: {sentence}")
        sentence = sentence.strip()
        # print(f"sentence-match5-strip: {sentence}")
        return sentence, score
    else:
        print('Illegal dataset name!')
        return None


# Select the content in the last quotation mark/bolded symbol ** : 'XXX'|"XXX"|\'XXX\'|\"XXX\"|**XXX** [Filtering score +1]
# Exclude single quotes in omitting such as's n't
def paired_filter(sentence, task_name):
    if task_name in ['vqav2', 'gqa', 'sqa', 'textvqa']:
        score = 0
        # Find the position of single quotes in abbreviations (such as can't, it's, etc.)
        shorthand_positions = set(
            match.start() + 1 for match in re.finditer(r"[a-zA-Z]'[a-zA-Z]", sentence)
        )

        # Match all the contents wrapped in quotation marks/asterisks
        pattern = re.finditer(r"('([^']+)'|\"([^\"]+)\"|\\'([^']+)\\'|\\\"([^\"]+)\\\"|\*\*([^*]+)\*\*)", sentence)

        valid_matches = []
        for match in pattern:
            span = match.span()
            quote_positions = {span[0], span[1] - 1}
            # Skip if it is a single quote in an abbreviation (that is, the position of the quote is within the abbreviation)
            if quote_positions & shorthand_positions:
                continue
            valid_matches.append(match)

        if valid_matches:
            last_match = valid_matches[-1]
            # Take the non-empty group
            for group in reversed(last_match.groups()):
                if group:
                    score = 1
                    sentence = group
                    break

    elif task_name == 'pope':
        score = 0
         # Find the position of single quotes in abbreviations (such as can't, it's, etc.)
        shorthand_positions = set(
            match.start() + 1 for match in re.finditer(r"[a-zA-Z]'[a-zA-Z]", sentence)
        )

        # Find the position where "* "appears
        special_chars = {"'", '"', '*'}
        all_indices = set(i for i, ch in enumerate(sentence) if ch in special_chars) 

        # Remove punctuation marks that are not abbreviated
        final_indices = all_indices - shorthand_positions
        sentence = ''.join([ch for i, ch in enumerate(sentence) if i not in final_indices])
        if len(final_indices):
            score = 1

    sentence = custom_strip(sentence)
    return sentence, score


# Make sure there are no extra Spaces or punctuation marks in the answer, except for TextVQA
def final_strip(final_text, task_name):
    if task_name in ['vqav2', 'gqa', 'pope', 'sqa']:
        final_text = re.sub(r"^[\s'\"()\[\]\{\}<>\-.,;:!?]+|[\s'\"()\[\]\{\}<>\-.,;:!?]+$", "", final_text)
    return final_text.strip()      


# Get the last sentence and remove the blanks. [for QA-SFT]
def get_last_sentence(text):
    text = text.replace('\n', ' ')
    sentence_lst = split_sentences(text)
    final_answer = sentence_lst[-1].strip()
    return final_answer


########## The main function for filtering the answers ##########
def extract_answer(text, filter_answer, split_word='ASSISTANT', task_name='vqav2', model_type='zero-shot'):
    # print('\n')   
    # Step 1: Extract the content following the last appearing segmentation word (such as ASSISTANT, assistant, <|assistant|>, <|ASSISTANT|>, or helper, etc.).
    text = extract_split_word(text, split_word)
    # print("after pattern1:", [text])

    # The model fine-tuned by QA-SFT takes the last sentence as the answer without any other complex processing
    if model_type == 'qa-sft':
        text = get_last_sentence(text)

    if model_type == 'json_type':
        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

        if "\\n" in cleaned:
            cleaned = cleaned.replace("\\n", "\n")

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                answer = parsed.get("Answer") or parsed.get("answer")
                if answer is not None:
                    text = str(answer).strip()
                else:
                    print(f"[Info] Valid JSON but missing 'Answer' key. Raw text: {json_str[:100]}...")
            except json.JSONDecodeError as e:
                print(f"[Warn] JSON decode failed: {e}. Using raw text {text} as fallback.")
        else:
            print(f"[Info] No JSON object found in text: {text[:100]}...")


    if not filter_answer:
        return text

    # Ignore the influence of line breaks: Replace \n with a space (without affecting the result structure)
    text = text.replace('\n', ' ')

    if task_name in ['vqav2', 'gqa', 'pope', 'sqa']:
        # Step 2: Divide the complete answer into several sentences with the delimiter being...; ? ! And the ". "in the quotation marks" '\" \'" ? ! Not divided.
        sentence_lst = split_sentences(text)

        # Step 3: Filter in sequence using different modes
        # vqav2, gqa: Handle all kinds of punctuation marks
        result_lst = []
        score_dict = {}
        for idx, sentence in enumerate(sentence_lst):
            # print(idx, sentence)
            # Statistical filtering scores
            score_dict[idx] = 0
            # 1. Core Filtering Mode [Filtering core +2]
            sentence, score = core_filter(sentence, task_name)
            score_dict[idx] += score
            # print(sentence, score)
            # 2. Special Filtering Mode [Filtering core +2]
            sentence, score = special_filter(sentence, task_name)
            score_dict[idx] += score
            # print(sentence, score)
            # 3. Deal with inverted sentences: (,) XXX is ... the ... answer|response ... [Filtering score +2]
            sentence, score = inverted_filter(sentence, task_name)
            score_dict[idx] += score
            # print(sentence, score)
            # 4. Select the content in the last quotation mark/bolded symbol ** : 'XXX'|"XXX"|\'XXX\'|\"XXX\"|**XXX** [Filtering score +1]
            sentence, score = paired_filter(sentence, task_name)
            score_dict[idx] += score
            # print(sentence)
            # Save the final result
            result_lst.append(sentence)
        # print(f"score_dict: {score_dict}")
        # print(f"result_lst: {result_lst}")

        # Step 4: Choose the answer from several sentences
        # Try to find the sentences with high filtering scores and those at the bottom.
        # Sentences that appear later with the same score will directly overwrite the originally stored answers.
        final_text = ""
        final_score = -1
        for idx, score in score_dict.items():
            if score >= final_score:
                final_score = score
                final_text = result_lst[idx]
        if task_name == 'pope' and final_score == 0:
            final_text = ' '.join(result_lst)
        
        # Step 5: After removing the blank Spaces and English punctuation at both ends of the text, return
        final_text = final_strip(final_text, task_name)
    elif task_name == 'textvqa':
        # textvqa: Not dividing sentences. Scores are not counted.
        # 1. Core Filtering Mode
        text, _ = core_filter(text, task_name)
        # 2. Special Filtering Mode
        text, _ = special_filter(text, task_name)
        # 3. Deal with inverted sentences: XXX is the ... answer|response ...
        text, _ = inverted_filter(text, task_name)
        # 4. Select the content in the last quotation mark/bolded symbol ** : 'XXX'|"XXX"|\'XXX\'|\"XXX\"|**XXX**
        text, _ = paired_filter(text, task_name)
        # 5. Return directly after removing the whitespace characters
        final_text = final_strip(text, task_name)
    else:
        print('Illegal dataset name!')
        return None

    return final_text


if __name__ == "__main__":
    ############################## Test samples ##############################
    ##############################   Group 1    ##############################
    print("\n\nGroup 1:")
    # 1. Test sample for the extract_split_word function [equivalent to filter_answer=False, directly calling the main function]
    # All the answers are in the following form:
    # x_y yes, it is. / x_y No, it isn't.
    # Sample 1: A single assistant, all in lowercase (with colon/without colon)
    text1_1 = "user: \n\nis it a cat? \n assistant 1_1 yes, it is."
    print([extract_answer(text1_1, False)])
    text1_2 = "user: \n\nis it a cat? \n assistant: 1_2 yes, it is."
    print([extract_answer(text1_2, False)])

    # Sample 2: A single ASSISTANT (with colon/without colon)
    text2_1 = "USER: \n\nIs it a dog? \n ASSISTANT 2_1 yes, it is."
    print([extract_answer(text2_1, False)])
    text2_2 = "USER: \n\nIs it a dog? \n ASSISTANT: 2_2 yes, it is."
    print([extract_answer(text2_2, False)])

    # Sample 3: A single <|ASSISTANT|> (with colon/without colon)
    text3_1 = "USER: \n\nIs it a bear? \n <|ASSISTANT|> 3_1 No, it isn't."
    print([extract_answer(text3_1, False)])
    text3_2 = "USER: \n\nIs it a bear? \n <|ASSISTANT|>: 3_2 No, it isn't."
    print([extract_answer(text3_2, False)])

    # Sample 4: A single <|assistant|> (with colon/without colon)
    text4_1 = "USER: \n\nIs it a snake? \n <|assistant|> 4_1 No, it isn't."
    print([extract_answer(text4_1, False)])
    text4_2 = "USER: \n\nIs it a snake? \n <|assistant|>: 4_2 No, it isn't."
    print([extract_answer(text4_2, False)])

    # Sample 5: assistant+ASSISTANT (with colon/Without colon)
    text5_1 = "ASSISTANT \nUSER: \n\nIs it a panda? \n assistant 5_1 No, it isn't."
    print([extract_answer(text5_1, False)])
    text5_2 = "ASSISTANT \nUSER: \n\nIs it a panda? \n assistant: 5_2 No, it isn't."
    print([extract_answer(text5_2, False)])
    text5_3 = "ASSISTANT: USER: \n\nIs it a panda? \n assistant 5_3 No, it isn't."
    print([extract_answer(text5_3, False)])
    text5_4 = "ASSISTANT: USER: \n\nIs it a panda? \n assistant: 5_4 No, it isn't."
    print([extract_answer(text5_4, False)])
    text5_5 = "assistant \nUSER: \n\nIs it a panda? \n ASSISTANT 5_5 No, it isn't."
    print([extract_answer(text5_5, False)])
    text5_6 = "assistant \nUSER: \n\nIs it a panda? \n ASSISTANT: 5_6 No, it isn't."
    print([extract_answer(text5_6, False)])
    text5_7 = "assistant: USER: \n\nIs it a panda? \n ASSISTANT 5_7 No, it isn't."
    print([extract_answer(text5_7, False)])
    text5_8 = "assistant: USER: \n\nIs it a panda? \n ASSISTANT: 5_8 No, it isn't."
    print([extract_answer(text5_8, False)])

    # Sample 6: <|ASSISTANT|> or <|assistant|>+ASSISTANT (with colon/without colon)
    text6_1 = "<|ASSISTANT|> \nUSER: \n\nIs it a wolf? \n ASSISTANT 6_1 No, it isn't."
    print([extract_answer(text6_1, False)])
    text6_2 = "<\|ASSISTANT\|> \nUSER: \n\nIs it a wolf? \n ASSISTANT: 6_2 No, it isn't."
    print([extract_answer(text6_2, False)])
    text6_3 = "<\|ASSISTANT\|>: USER: \n\nIs it a wolf? \n ASSISTANT 6_3 No, it isn't."
    print([extract_answer(text6_3, False)])
    text6_4 = "<|ASSISTANT|>: USER: \n\nIs it a wolf? \n ASSISTANT: 6_4 No, \'it isn't."
    print([extract_answer(text6_4, False)])
    text6_5 = "<|assistant|> \nUSER: \n\nIs it a snake? \n ASSISTANT 6_5 No, it isn't."
    print([extract_answer(text6_5, False)])
    text6_6 = "<\|assistant\|> \nUSER: \n\nIs it a snake? \n ASSISTANT: 6_6 No, it isn't."
    print([extract_answer(text6_6, False)])
    text6_7 = "<\|assistant\|>: USER: \n\nIs it a snake? \n ASSISTANT 6_7 No, it isn't."
    print([extract_answer(text6_7, False)])
    text6_8 = "<|assistant|>: USER: \n\nIs it a snake? \n ASSISTANT: 6_8 No, it isn't."
    print([extract_answer(text6_8, False)])

    # Example 7: Replace the segmentation words with HELPER/ANSWER/RESPONSE, etc
    text7_1 = "USER: \n\nIs it a sheep? \n ASSISTANT: 7_1 No, it isn't."
    print([extract_answer(text7_1, False, split_word='ASSISTANT')])
    text7_2 = "USER: \n\nIs it a sheep? \n HELPER: 7_2 No, \'it isn't.\'"
    print([extract_answer(text7_2, False, split_word='HELPER')])
    text7_3 = "USER: \n\nIs it a sheep? \n ANSWER: 7_3 No, it isn't."
    print([extract_answer(text7_3, False, split_word='ANSWER')])    
    text7_4 = "USER: \n\nIs it a sheep? \n RESPONSE: 7_4 No, it isn't."
    print([extract_answer(text7_4, False, split_word='RESPONSE')])  
    text7_5 = "How many apple are shown?\nAnswer the question using a single word or phrase. CAT: 0"
    print([extract_answer(text7_5, False, split_word='CAT')])

    ##############################  Group 2  ##############################
    print("\n\nGroup 2:")
    # 2. Test samples for the custom_strip function
    text_lst = [
        ' It is a cat\n', 'it\'s a dog\n.', '**it\'s a snake**', "\'\t That is bear's home.\n", '**\nI\'m panda', 
        '(dogs.)', '[blue]', '{green', r'{yellow}', '[(brown', "\'color", '\"cat, dog, pig, fish.', '"', "'", "'I am fine.",
        # Supplementary: The handling of punctuation in sentences, whether in pairs or not
        'It\'s a \'cat\'', 'This\'s a [cat\'', 'he\'s **Tom**.', 'she can\'t **move)!', 'she shouldn\'t **[{do it)}!',
        '\"tom\'', '"marry\"', "you\'re \'lucy\'", 'Jerry\'\"**]}[{)(".', "likely \"The luggage cart is pulled by a person wearing a vest.\"",
        "The text in the image reads: \"Who is the luggage cart pulled by? Answer the question using a single word or phrase.\"\n\nThis is a playful question, and the answer is likely \"The luggage cart is pulled by a person wearing a vest.\"",
    ]
    # ['It is a cat', "it's a dog\n.", "**it's a snake**", "That is bear's home.", "I'm panda",
    #  '(dogs.)', '[blue]', '{green', '{yellow}', '[(brown', 'color', 'cat, dog, pig, fish.', '', '', 'I am fine.',
    #  "It's a 'cat'", "This's a [cat", "he's **Tom**.", "she can't move)!", "she shouldn't [{do it)}!",
    #  'tom', '"marry"', "you're 'lucy'", 'Jerry"]}[{)(".', 'likely "The luggage cart is pulled by a person wearing a vest."',
    #  'The text in the image reads: "Who is the luggage cart pulled by? Answer the question using a single word or phrase."\n\nThis is a playful question, and the answer is likely "The luggage cart is pulled by a person wearing a vest."']
    ans_lst = []
    for text in text_lst:
        ans = custom_strip(text)
        print(f"final_answer: {[ans]}")
        ans_lst.append(ans)
    print(ans_lst)

    ##############################  Group 3  ##############################
    print("\n\nGroup 3:")    
    # 3. Test sample for the sentence segmentation function split_sentences(text)
    text_lst = [
        'USER: \n\n ASSISTANT: 1_1 The question is: \"Is it overcast?\" \nThe answer based on visual info is: \"Yes, it\'s.\"', # one sentence
        'USER: \n\n ASSISTANT: 1_2 The question is: \'Is it overcast?\', \nThe answer based on visual info is: "Yes, it\'s.".', # one sentence
        'USER: \n\n ASSISTANT: 1_3 The question is: "Is it overcast?". \nThe answer based on visual info is: "Yes."!', # two sentences
        'USER: \n\n ASSISTANT: 1_4 The question is: \'Is it overcast?\'! \nThe answer based on visual info is: \"Yes.\"?', # two sentences
        "USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?'; \nThe answer based on visual info is: \"Yes.\";", # two sentences
        # five sentences
        'The image features a traffic light displaying a red pedestrian signal, indicating that it is not safe for pedestrians to cross the street. The traffic light is mounted on a pole, and there is a sign below it with a question asking, "Which way is it to the museum? Answer the question using a single word or phrase."\n\nThe sign provides directions to various museums in the area, including Museum im Grassi, Schumann-Haus, Museum bildende Kunst, Stadtgeschicht, Museum, Zeitgeschicht, and Forum. The sign is written in German, and it is likely that the sign is located in a German-speaking region, as it provides information about museums in Germany.\n\nThe image also features a building with a large letter "E" on it, which could be a part of the building\'s architecture or a sign for a nearby establishment. The overall scene suggests that the image is taken in a public area, possibly near a museum or a place of interest, where pedestrians are required to follow the traffic signal and follow the directions provided by the sign.',
        ## Supplementary example: The last sentence is without punctuation
        "USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?'; \nThe answer based on visual info is: \"Yes.\"", # two sentences
        "USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?'. \nThe answer based on visual info is: \"Yes.\"", # two sentences
        "USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?'; \nThe answer based on visual info is: \"Yes.\"", # two sentences
        "USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?'; \nThe answer based on visual info is: \"Yes.\"", # two sentences
        "The text in the image reads: \"Who is the luggage cart pulled by? Answer the question using a single word or phrase.\"\n\nThis is a playful question, and the answer is likely \"The luggage cart is pulled by a person wearing a vest.\"",
        # Supplementary examples again: A. U.S., etc.
        "A., U.S., etc.", "A. dog B. cat C.snake D. 4.32", "A. dog", "U.S. 3.1415926 $1.9", "The question is: 'Is it overcast? A. yes B.no'. The answer is B. no"
    ]
    # output:
    # sentence_lst_len=1, sentence_lst: ['USER: \n\n ASSISTANT: 1_1 The question is: "Is it overcast?" \nThe answer based on visual info is: "Yes, it\'s."']
    # sentence_lst_len=1, sentence_lst: ['USER: \n\n ASSISTANT: 1_2 The question is: \'Is it overcast?\', \nThe answer based on visual info is: "Yes, it\'s.".']
    # sentence_lst_len=2, sentence_lst: ['USER: \n\n ASSISTANT: 1_3 The question is: "Is it overcast?".', 'The answer based on visual info is: "Yes."!']
    # sentence_lst_len=2, sentence_lst: ["USER: \n\n ASSISTANT: 1_4 The question is: 'Is it overcast?'!", 'The answer based on visual info is: "Yes."?']
    # sentence_lst_len=2, sentence_lst: ["USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?';", 'The answer based on visual info is: "Yes.";']
    # sentence_lst_len=5, sentence_lst: ['The image features a traffic light displaying a red pedestrian signal, indicating that it is not safe for pedestrians to cross the street.', 'The traffic light is mounted on a pole, and there is a sign below it with a question asking, "Which way is it to the museum? Answer the question using a single word or phrase."\n\nThe sign provides directions to various museums in the area, including Museum im Grassi, Schumann-Haus, Museum bildende Kunst, Stadtgeschicht, Museum, Zeitgeschicht, and Forum.', 'The sign is written in German, and it is likely that the sign is located in a German-speaking region, as it provides information about museums in Germany.', 'The image also features a building with a large letter "E" on it, which could be a part of the building\'s architecture or a sign for a nearby establishment.', 'The overall scene suggests that the image is taken in a public area, possibly near a museum or a place of interest, where pedestrians are required to follow the traffic signal and follow the directions provided by the sign.']
    
    # sentence_lst_len=2, sentence_lst: ["USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?';", 'The answer based on visual info is: "Yes."']
    # sentence_lst_len=2, sentence_lst: ["USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?'.", 'The answer based on visual info is: "Yes."']
    # sentence_lst_len=2, sentence_lst: ["USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?';", 'The answer based on visual info is: "Yes."']
    # sentence_lst_len=2, sentence_lst: ["USER: \n\n ASSISTANT: 1_5 The question is: 'Is it overcast?';", 'The answer based on visual info is: "Yes."']
    # sentence_lst_len=1, sentence_lst: ['The text in the image reads: "Who is the luggage cart pulled by? Answer the question using a single word or phrase."\n\nThis is a playful question, and the answer is likely "The luggage cart is pulled by a person wearing a vest."']
    
    # Handle special abbreviations:
    # sentence_lst_len=1, sentence_lst: ['A., U.S., etc.']
    # sentence_lst_len=1, sentence_lst: ['A. dog B. cat C.snake D. 4.32']
    # sentence_lst_len=1, sentence_lst: ['A. dog']
    # sentence_lst_len=1, sentence_lst: ['U.S. 3.1415926 $1.9']
    # sentence_lst_len=2, sentence_lst: ["The question is: 'Is it overcast? A. yes B.no'.", 'The answer is B. no']
    # Do not handle special abbreviations:
    # sentence_lst_len=1, sentence_lst: ['A., U.S., etc.']
    # sentence_lst_len=3, sentence_lst: ['A.', 'dog B.', 'cat C.snake']
    # sentence_lst_len=2, sentence_lst: ['A.', 'dog']
    # sentence_lst_len=1, sentence_lst: ['U.S.']
    # sentence_lst_len=3, sentence_lst: ["The question is: 'Is it overcast? A. yes B.no'.", 'The answer is B.', 'no']
    for text in text_lst:
        # print(f"split sentence: {split_sentences(text)}")
        split_sentences(text)

    ##############################  Group 4  ##############################
    print("\n\nGroup 4:")
    # 4. Test the function core_filter(sentence) for the core matching pattern
    text_lst = [
        'The answer to the question is a cat.', 'The correct response is woman.', 'The true answer should be: man', "Therefore, answer could be: panda.",
        'but the answer provided could be: panda\'s home.', 'and the answer provided would be yes, which means it\'s overcast', 
        'the response is likely:    snake', 'Response is: a cat.', 'Answer is: woman.', 'response: man.', 'Answer: man.', "but answer is: panda.",
        'the answer is likely "The luggage cart is pulled by a person wearing a vest."',
        # Supplementary: Observe the influence of time on the results
        'the answer is 8:20', 'the answer is: 8:20', 'the answer is : 8:20', 'the answer is :8:20', 'the answer is:8:20'
    ]
    # sqa:
    # [('a cat.', 2), ('woman.', 2), ('man', 2), ('panda.', 2),
    #  ("panda's home.", 2), ('yes', 2),
    #  ('snake', 2), ('a cat.', 2), ('woman.', 2), ('response: man.', 0), ('Answer: man.', 0), ('panda.', 2),
    #  ('likely "The luggage cart is pulled by a person wearing a vest."', 2),
    #  ('8:20', 2), ('8:20', 2), ('8:20', 2), (':8:20', 2), (':8:20', 2)]

    # pope:
    # [('a cat.', 2), ('woman.', 2), ('man', 2), ('panda.', 2),
    #  ("panda's home.", 2), ("yes, which means it's overcast", 2),
    #  ('snake', 2), ('a cat.', 2), ('woman.', 2), ('response: man.', 0), ('Answer: man.', 0), ('panda.', 2),
    #  ('likely "The luggage cart is pulled by a person wearing a vest."', 2),
    #  ('8:20', 2), ('8:20', 2), ('8:20', 2), (':8:20', 2), (':8:20', 2)]
    for task_name in ['sqa', 'pope']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text_lst:
            ans = core_filter(text, task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)

    ##############################  Group 5  ##############################
    print("\n\nGroup 5:")
    # 5. Test the function special_filter(sentence) for the special matching pattern
    text_lst = [
        'Therefore, answer: a cat.', 'Answer:\n\n woman, not man.', 'The response :man.', "So response: panda.",
        'Therefore, answer :\n\n a cat.', 'Answer  :\n\n woman, not man.', 'Response :man.', 'response: panda\'s home.',
        # Supplementary: Observe the influence of time on the results
        'answer : 8:20', 'response: 8:20', 'answer:8:20', 'answer :8:20'
    ]
    # sqa:
    # [('a cat.', 2), ('woman', 2), ('The response :man.', 0), ('panda.', 2),
    #  ('a cat.', 2), ('woman', 2), ('Response :man.', 0), ("panda's home.", 2),
    #  ('8:20', 2), ('8:20', 2), ('answer:8:20', 0), ('answer :8:20', 0)]
    # pope:
    # [('a cat.', 2), ('woman, not man.', 2), ('The response :man.', 0), ('panda.', 2),
    #  ('a cat.', 2), ('woman, not man.', 2), ('Response :man.', 0), ("panda's home.", 2),
    #  ('8:20', 2), ('8:20', 2), ('answer:8:20', 0), ('answer :8:20', 0)]
    for task_name in ['sqa', 'pope']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text_lst:
            ans = special_filter(text, task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)

    ##############################  Group 6  ##############################
    print("\n\nGroup 6:")
    # 6. Test the function inverted_filter(sentence) for the special matching pattern
    text_lst = [
        'Therefore, a cat is the answer to the question.', 'A dog is likely the answer to the question.', 
        '"Panda" is the answer, not "snake".', 'Man is the correct answer to the question.', 
        '"Panda\'s home" is the answer, not "snake".', 'Man is likely the correct answer to the question.', 
    ]
    # sqa:
    # [('a cat', 2), ('A dog', 2), ('"Panda"', 2),
    #  ('Man', 2), ('"Panda\'s home"', 2), ('Man', 2)]
    # pope:
    # [('Therefore, a cat', 2), ('A dog', 2), ('"Panda"', 2),
    #  ('Man', 2), ('"Panda\'s home"', 2), ('Man', 2)]
    for task_name in ['sqa', 'pope']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text_lst:
            ans = inverted_filter(text, task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)


    ##############################  Group 7  ##############################
    print("\n\nGroup 7:")
    # 7. Test the function paired_filter(sentence) that selects the content within the last quotation mark/bolded symbol **/ parenthesis
    text_lst = [
        '"banana" and \'apple\'', '"Panda" is the answer, not "snake".', '\'apple\'', "\"banana\"", '"pine"', "'pineapple'",
        r'(banana) and {apple}', '"Panda" is the answer, not [snake].', '(apple)', "[banana]", r'{pine}', "**pineapple**", 
        '(banana) and *apple*', "**Panda** is the answer, not it\'s snake.", "\"banana\" and **apple**", 
        # Supplementary: Examples of asymmetrical symbols
        '(banana) and apple**', "**Panda* is the answer", "\"banana\" and (apple**"
    ]
    # sqa:
    # [('apple', 1), ('snake', 1), ('apple', 1), ('banana', 1), ('pine', 1), ('pineapple', 1),
    #  ('(banana) and {apple}', 0), ('Panda', 1), ('(apple)', 0), ('[banana]', 0), ('{pine}', 0), ('pineapple', 1),
    #  ('(banana) and *apple*', 0), ('Panda', 1), ('apple', 1),
    #  ('(banana) and apple', 0), ('Panda* is the answer', 0), ('banana', 1)]
    # pope:
    # [('banana and apple', 1), ('Panda is the answer, not snake.', 1), ('apple', 1), ('banana', 1), ('pine', 1), ('pineapple', 1),
    #  ('(banana) and {apple}', 0), ('Panda is the answer, not [snake].', 1), ('(apple)', 0), ('[banana]', 0), ('{pine}', 0), ('pineapple', 1),
    #  ('(banana) and apple', 1), ("Panda is the answer, not it's snake.", 1), ('banana and apple', 1),
    #  ('(banana) and apple', 1), ('Panda is the answer', 1), ('banana and (apple', 1)]
    for task_name in ['sqa', 'pope']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text_lst:
            ans = paired_filter(text, task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)

    ##############################  Group 8  ##############################
    print("\n\nGroup 8:")
    # 8. Test for final_strip(sentence)
    text_lst = [
        ':"an apple."', "orange,.?", '\'pineapple\'', 'stawberry?!.',
        "a bear.", "\ta bear,", "\na bear:", "a bear!", "a bear?", "'a bear'", '\'a bear\'', "a bear;"
    ]
    # pope:
    # ['an apple', 'orange', 'pineapple', 'stawberry',
    #  'a bear', 'a bear', 'a bear', 'a bear', 'a bear', 'a bear', 'a bear', 'a bear']
    # textvqa:
    # [':"an apple."', 'orange,.?', "'pineapple'", 'stawberry?!.',
    #  'a bear.', 'a bear,', 'a bear:', 'a bear!', 'a bear?', "'a bear'", "'a bear'", 'a bear;']
    for task_name in ['pope', 'textvqa']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text_lst:
            ans = final_strip(text, task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)

    ##############################  Group 9  ##############################
    print("\n\nGroup 9:")
    # Simple sentence test: If the test content cannot be filtered, has it been filtered incorrectly
    # First: Test whether phrases that do not require any filtering at all or only require punctuation are completely retained
    text1_lst = ["Bananas", "A cat.", "wonderful!", "Yes,", "'Train?'", "\'Val;\'", '\"Mary\'s cat.\"']
    # sqa/pope:
    # ['Bananas', 'A cat', 'wonderful', 'Yes', 'Train', 'Val', "Mary's cat"]
    # textvqa:
    # ['Bananas', 'A cat.', 'wonderful!', 'Yes,', 'Train?', 'Val;', "Mary's cat."]
    for task_name in ['sqa', 'pope', 'textvqa']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text1_lst:
            ans = extract_answer(text, True, 'ASSISTANT', task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)

    # Second: Check whether the sentences in the test content that cannot be filtered are completely retained
    text2_lst = [
        "The toaster is not to the right of a refrigerator; \nit's actually in front of one.",
        # Sentences containing distractors
        "Who is that spectator watching? The answer the question using a single word or phrase. \nIt is a cat.", # There exists "the answer" and the following "is", which should not match in the two sentences
        'it\'s a word, Answer the question using a single word or phrase.\n', # Escape characters with's 'should be completely retained
    ]
    # sqa:
    # ["it's actually in front of one", 'It is a cat', "it's a word, Answer the question using a single word or phrase"]
    # pope:
    # ["The toaster is not to the right of a refrigerator; it's actually in front of one",
    #  'Who is that spectator watching? The answer the question using a single word or phrase. It is a cat',
    #  "it's a word, Answer the question using a single word or phrase"]
    # textvqa:
    # ["The toaster is not to the right of a refrigerator;  it's actually in front of one.",
    #  'Who is that spectator watching? The answer the question using a single word or phrase.  It is a cat.',
    #  "it's a word, Answer the question using a single word or phrase."]
    
    for task_name in ['sqa', 'pope', 'textvqa']:
        ans_lst = []
        print("task_name:", task_name, '\n')
        for text in text2_lst:
            ans = extract_answer(text, True, 'ASSISTANT', task_name)
            print(f"final_answer: {[ans]}")
            ans_lst.append(ans)
        print(ans_lst)

    ##############################  Group 10  ##############################
    print("\n\nGroup 10:")
    # Complex sentence test: 58 real outputs based on reasoning + the original 9 long sentence examples to view the implementation effect of the filtering function
    for task_name in ['sqa', 'textvqa', 'pope']:
        print("task_name:", task_name, '\n')
        text1 = "Bananas"
        print(f"final_answer: {[extract_answer(text1, True, 'ASSISTANT', task_name)]}") # ['Bananas']

        text2 = "The toaster is not to the right of a refrigerator; it's actually in front of one." 
        print(f"final_answer: {[extract_answer(text2, True, 'ASSISTANT', task_name)]}") # ["it's actually in front of one"]
        # ["The toaster is not to the right of a refrigerator; it's actually in front of one."]
        # ["The toaster is not to the right of a refrigerator; it's actually in front of one"]

        text3 = "Who is that spectator watching? Answer the question using a single word or phrase." 
        print(f"final_answer: {[extract_answer(text3, True, 'ASSISTANT', task_name)]}") # ['Answer the question using a single word or phrase']
        # ['Who is that spectator watching? Answer the question using a single word or phrase.']
        # ['Who is that spectator watching? Answer the question using a single word or phrase']

        text4 = "The word \"firenze\" is the answer to what is underneath the covered table." 
        print(f"final_answer: {[extract_answer(text4, True, 'ASSISTANT', task_name)]}") # ['underneath the covered table']
        # textvqa: ['underneath the covered table.']

        text5 = "Answer: curtains"
        print(f"final_answer: {[extract_answer(text5, True, 'ASSISTANT', task_name)]}") # ['curtains']

        text6 = "The pizza in the image is topped with spinach, which is a common topping for pizzas. The question asks for the name of the food with spinach, and the answer is a pizza."
        print(f"final_answer: {[extract_answer(text6, True, 'ASSISTANT', task_name)]}")  # ['a pizza']
        # textvqa: ['a pizza.']

        text7 = "The text in the image reads: \"Who is the luggage cart pulled by? Answer the question using a single word or phrase.\"\n\nThis is a playful question, and the answer is likely \"The luggage cart is pulled by a person wearing a vest.\""
        print(f"final_answer: {[extract_answer(text7, True, 'ASSISTANT', task_name)]}") # ['The luggage cart is pulled by a person wearing a vest']
        # ['The luggage cart is pulled by a person wearing a vest.']
        # ['likely The luggage cart is pulled by a person wearing a vest']

        text8 = "The text in the image reads: \"Is there either a blue bed or sofa? Answer the question using a single word or phrase.\"\n\nThis is a playful question, and the answer could be either a blue bed or a blue sofa."
        print(f"final_answer: {[extract_answer(text8, True, 'ASSISTANT', task_name)]}") # ['either a blue bed or a blue sofa']
        # textvqa: ['either a blue bed or a blue sofa.']

        text9 = "The question asks which color the horse is, and the answer is \"brown\" or \"tan.\""
        print(f"final_answer: {[extract_answer(text9, True, 'ASSISTANT', task_name)]}") # ['tan']
        # ['tan.']
        # ['brown or tan']

        text10 = "The question is asking which kind of furniture is long. The answer is a single word or phrase: \"bed.\""
        print(f"final_answer: {[extract_answer(text10, True, 'ASSISTANT', task_name)]}") # ['bed']
        # textvqa: ['bed.']

        text11 = "The question is: \"Do the trees in front of the fence look green and abundant? Answer the question using a single word or phrase.\"\n\nIn this context, the answer would be \"Yes,\" as the trees in front of the fence appear to be green and abundant."
        print(f"final_answer: {[extract_answer(text11, True, 'ASSISTANT', task_name)]}") # ['Yes']
        # ['Yes,']
        # ['Yes, as the trees in front of the fence appear to be green and abundant']

        text12 = "The image shows two elephants and a dolphin. The question asks if there are both an elephant and a dolphin in the picture, and the answer is no, as there is only one elephant and one dolphin in the picture."
        print(f"final_answer: {[extract_answer(text12, True, 'ASSISTANT', task_name)]}") # ['no']
        # ['no, as there is only one elephant and one dolphin in the picture.']
        # ['no, as there is only one elephant and one dolphin in the picture']

        text13 = "The image shows a dessert setup with two glasses of milkshake and a slice of cake. The question asks which shape the woman is behind the food, and the answer is based on a single word or phrase. The correct answer is \"octagon.\""
        print(f"final_answer: {[extract_answer(text13, True, 'ASSISTANT', task_name)]}") # ['octagon']
        # textvqa: ['octagon.']

        text14 = "The sheep in the image are wearing red tags, which are commonly used to identify them for tracking and management purposes. The question about the color of the jacket is a playful one, and the answer can be given in a single word or phrase: \"red.\""
        print(f"final_answer: {[extract_answer(text14, True, 'ASSISTANT', task_name)]}") # ['red']
        # ['red.']
        # ['The question about the color of the jacket is a playful one, and the answer can be given in a single word or phrase: red']

        text15 = "The text in the image reads: \"Is there any red table or bottle? Answer the question using a single word or phrase.\"\n\nThis is a playful riddle where the answer is \"red table\" because the person is pouring a red liquid into a glass."
        print(f"final_answer: {[extract_answer(text15, True, 'ASSISTANT', task_name)]}") # ['red table']
        # pope: ['red table because the person is pouring a red liquid into a glass']

        text16 = "The motorcycle in the photo is a vintage or classic model, possibly a Triumph or a similar brand. It has a camouflage-patterned bag on the back, which is a common feature in vintage motorcycles. The text in the image is asking for a single word or phrase to describe the motorcycle, but the answer provided is \"motorbike,\" which is a bit vague. A more specific description would be \"classic motorcycle.\""
        print(f"final_answer: {[extract_answer(text16, True, 'ASSISTANT', task_name)]}") # ['motorbike']
        # ['classic motorcycle.']
        # ['motorbike, which is a bit vague']

        text17 = "The food in the image is a lunchbox with a variety of items. Here's a breakdown of the contents:\n\n1. **Carrot Cubes**: These are small, cubed pieces of carrot.\n2. **Green Soybeans**: These are oval-shaped green beans.\n3. **Oreo Cookies**: There are several cookies in the container.\n4. **Muffin Cups**: There are two muffin cups, one with colorful sprinkles and the other with white rice.\n5. **Sushi Rolls**: There are two sushi rolls, one with white rice and the other with seaweed and sesame seeds.\n\nThe question in the image is asking which type of food is not sprinkled, and the answer is the muffin cups, as they do not have any visible sprinkles."
        print(f"final_answer: {[extract_answer(text17, True, 'ASSISTANT', task_name)]}") # ['the muffin cups']
        # ['the muffin cups, as they do not have any visible sprinkles.']
        # ['the muffin cups, as they do not have any visible sprinkles']

        text18 = "Question: How many suitcases are on the baggage cart? Answer the question using a single word or phrase.\n\nAnswer: There are four suitcases on the baggage cart."
        print(f"final_answer: {[extract_answer(text18, True, 'ASSISTANT', task_name)]}") # ['There are four suitcases on the baggage cart']
        # textvqa: ['There are four suitcases on the baggage cart.']

        text19 = "Question: What is sitting on top of the white drawer? Answer the question using a single word or phrase.\n\nAnswer: A potted plant is sitting on top of the white drawer."
        print(f"final_answer: {[extract_answer(text19, True, 'ASSISTANT', task_name)]}") # ['A potted plant is sitting on top of the white drawer']
        # textvqa: ['A potted plant is sitting on top of the white drawer.']

        text20 = "Question: What color is the horse? Answer the question using a single word or phrase.\n\nAnswer: The horse is brown."
        print(f"final_answer: {[extract_answer(text20, True, 'ASSISTANT', task_name)]}") # ['The horse is brown']
        # textvqa: ['The horse is brown.']

        text21 = "Question: Where is the toaster? Answer the question using a single word or phrase.\n\nAnswer: The toaster is in front of the refrigerator."
        print(f"final_answer: {[extract_answer(text21, True, 'ASSISTANT', task_name)]}") # ['The toaster is in front of the refrigerator']
        # textvqa: ['The toaster is in front of the refrigerator.']

        text22 = "Question: Who is the luggage cart pulled by? Answer the question using a single word or phrase.\n\nAnswer: The luggage cart is pulled by a man in a vest."
        print(f"final_answer: {[extract_answer(text22, True, 'ASSISTANT', task_name)]}") # ['The luggage cart is pulled by a man in a vest']
        # textvqa: ['The luggage cart is pulled by a man in a vest.']

        text23 = "Question: Is there either a blue bed or sofa? Answer the question using a single word or phrase.\n\nAnswer: Yes."
        print(f"final_answer: {[extract_answer(text23, True, 'ASSISTANT', task_name)]}") # ['Yes']
        # textvqa: ['Yes.']

        text24 = "Question: Do the trees in front of the fence look green and abundant? Answer the question using a single word or phrase.\n\nAnswer: Yes."
        print(f"final_answer: {[extract_answer(text24, True, 'ASSISTANT', task_name)]}") # ['Yes']
        # textvqa: ['Yes.']

        text25 = "Question: Are there both an elephant and a dolphin in the picture? Answer the question using a single word or phrase.\n\nAnswer: No."
        print(f"final_answer: {[extract_answer(text25, True, 'ASSISTANT', task_name)]}") # ['No']
        # textvqa: ['No.']

        text26 = "Question: Which shape is the woman behind the food? Answer the question using a single word or phrase.\n\nAnswer: Octagon."
        print(f"final_answer: {[extract_answer(text26, True, 'ASSISTANT', task_name)]}") # ['Octagon']
        # textvqa: ['Octagon.']

        text27 = "Question: What color is the jacket? Answer the question using a single word or phrase.\n\nAnswer: Red."
        print(f"final_answer: {[extract_answer(text27, True, 'ASSISTANT', task_name)]}") # ['Red']
        # textvqa: ['Red.']

        text28 = "Question: Is there any red table or bottle? Answer the question using a single word or phrase.\n\nAnswer: Red table."
        print(f"final_answer: {[extract_answer(text28, True, 'ASSISTANT', task_name)]}") # ['Red table']
        # textvqa: ['Red table.']

        text29 = "Question: What type of vehicle is this? Answer the question using a single word or phrase.\n\nAnswer: Classic motorcycle."
        print(f"final_answer: {[extract_answer(text29, True, 'ASSISTANT', task_name)]}") # ['Classic motorcycle']
        # textvqa: ['Classic motorcycle.']

        text30 = "Question: Which type of food is not sprinkled? Answer the question using a single word or phrase.\n\nAnswer: Muffin cups."
        print(f"final_answer: {[extract_answer(text30, True, 'ASSISTANT', task_name)]}") # ['Muffin cups']
        # textvqa: ['Muffin cups.']

        text31 = "The image features a yellow pedestrian crossing sign with a man walking on it. The sign is located near a palm tree, and there are several other palm trees in the background. The sign is placed on a pole, and the surrounding area appears to be a beach or a tropical location. The sign is accompanied by a caption, which asks the question, \"Which kind of sign is this?\", and provides the answer, \"Pedestrian crossing sign.\""
        print(f"final_answer: {[extract_answer(text31, True, 'ASSISTANT', task_name)]}") # ['Pedestrian crossing sign']
        # ['Pedestrian crossing sign.']
        # ['The sign is accompanied by a caption, which asks the question, Which kind of sign is this?, and provides the answer, Pedestrian crossing sign']

        text32 = '''The person wearing the wristband is the girl in the middle.'''
        print(f"final_answer: {[extract_answer(text32, True, 'ASSISTANT', task_name)]}") # ['The person wearing the wristband is the girl in the middle']
        # textvqa: ['The person wearing the wristband is the girl in the middle.']

        text33 =  "To determine the appliance that reflects the miniature house, we need to identify the reflective surface in the image. The miniature house is clearly reflected on the surface of the toaster.\n\nTherefore, the answer is:\n\nToaster"
        print(f"final_answer: {[extract_answer(text33, True, 'ASSISTANT', task_name)]}") # ['Toaster']

        text34 = "The person is wearing a red jacket, which suggests they are likely an adult. Given the context of the image, it's reasonable to infer that the person is middle-aged or older. Therefore, the answer to the question \"How old is the person that the jacket is worn around?\" using a single word or phrase would be:\n\nMiddle-aged"
        print(f"final_answer: {[extract_answer(text34, True, 'ASSISTANT', task_name)]}") # ['How old is the person that the jacket is worn around']
        # ['How old is the person that the jacket is worn around?']
        # ['Therefore, the answer to the question How old is the person that the jacket is worn around? using a single word or phrase would be:  Middle-aged']

        text35 = "The sun shining through the umbrella is the answer to the question \"What is the sun shining through?\" using a single word or phrase."
        print(f"final_answer: {[extract_answer(text35, True, 'ASSISTANT', task_name)]}") # ['The sun shining through the umbrella']

        text36 =  "Yes, the countertop above the chairs that are next to the cupboard is the answer to the question using a single word or phrase."
        print(f"final_answer: {[extract_answer(text36, True, 'ASSISTANT', task_name)]}") # sqa: ['the countertop above the chairs that are next to the cupboard']
        # ['Yes, the countertop above the chairs that are next to the cupboard']

        text37 = "The person on the left of the image is standing on the beach, not in the water. Therefore, the answer to the question \"Where does the person on the left of the image stand?\" is:\n\n**On the beach.**"
        print(f"final_answer: {[extract_answer(text37, True, 'ASSISTANT', task_name)]}") # ['On the beach']
        # ['On the beach.']
        # ['Therefore, the answer to the question Where does the person on the left of the image stand? is:  On the beach']

        text38 = "No, the forest behind an elephant is not the answer to the question \"Is the forest behind an elephant?\" The correct answer would be \"No,\""
        print(f"final_answer: {[extract_answer(text38, True, 'ASSISTANT', task_name)]}") # ['No']
        # textvqa: ['No,']

        text39 = "Yes, the tall man behind the door is the answer to the question using a single word or phrase."
        print(f"final_answer: {[extract_answer(text39, True, 'ASSISTANT', task_name)]}") # sqa: ['the tall man behind the door']
        # ['Yes, the tall man behind the door']

        text40 =  "The image presents a clear plastic container filled with a variety of food items. The container is divided into four compartments, each containing a different type of food. \n\nIn the top left compartment, there are orange carrot sticks. The top right compartment contains green edamame beans. The bottom left compartment holds a white cupcake adorned with green and red sprinkles. Lastly, the bottom right compartment contains a black sushi roll with white rice and sesame seeds.\n\nThe container also features a label with a question: \"Which kind of vegetable is to the right of the carrots?\" The answer to this question is \"carrots"
        print(f"final_answer: {[extract_answer(text40, True, 'ASSISTANT', task_name)]}") # ['carrots']

        text41 = "Answer is yes."
        print(f"final_answer: {[extract_answer(text41, True, 'ASSISTANT', task_name)]}") # ['yes']
        # textvqa: ['yes.']

        text42 = "Answer: Spinach"
        print(f"final_answer: {[extract_answer(text42, True, 'ASSISTANT', task_name)]}") # ['Spinach']

        text43 = "The image shows a woman brushing her teeth in a bathroom. There are two toothbrushes and two mats in the picture. The question posed is whether the toothbrushes are in the same picture as the mats. The answer to the question is yes, the toothbrushes and mats are in the same picture."
        print(f"final_answer: {[extract_answer(text43, True, 'ASSISTANT', task_name)]}") # ['yes']
        # ['yes, the toothbrushes and mats are in the same picture.']
        # ['yes, the toothbrushes and mats are in the same picture']

        text44 =  "The text asks if the utensil on top of the table looks clean and black. The visual shows a plate with various food items including mashed potatoes, carrots, corn, and what appears to be a bread roll or biscuit. There's also another dish visible in the background that seems to contain meat. No utensils can be seen directly on the table surface; however, there might be one partially out of frame at the edge of the photo. Therefore, the correct response would be:\n\nAnswer: No, the utensil on top of the table does not look clean and black."
        print(f"final_answer: {[extract_answer(text44, True, 'ASSISTANT', task_name)]}") # ['No']
        # ['No, the utensil on top of the table does not look clean and black.']
        # ['No, the utensil on top of the table does not look clean and black']

        text45 = "The question in the image is: \"Is the surfer that looks wet wearing a wetsuit? Answer the question using a single word or phrase.\"\n\nThe answer to the question is: Yes."
        print(f"final_answer: {[extract_answer(text45, True, 'ASSISTANT', task_name)]}") # ['Yes']
        # textvqa: ['Yes.']

        text46 = "The text within the image asks if there are any drapes to the right of the bed. The response provided by the user is \"Answer: Yes.\" This indicates that the person who took the photo has placed curtains or drapes on the right side of the bed for privacy or decoration purposes."
        print(f"final_answer: {[extract_answer(text46, True, 'ASSISTANT', task_name)]}") # ['Yes. This indicates that the person who took the photo has placed curtains or drapes on the right side of the bed for privacy or decoration purposes']
        # textvqa: ['Yes. This indicates that the person who took the photo has placed curtains or drapes on the right side of the bed for privacy or decoration purposes.']

        text47 = "The question in the image is: \"Are there drapes to the right of the bed? Answer the question using a single word or phrase.\"\n\nThe answer based on the visual information is: \"Yes.\""
        print(f"final_answer: {[extract_answer(text47, True, 'ASSISTANT', task_name)]}") # ['Yes']
        # textvqa: ['Yes.']

        text48 = "What is the question?\n\nAnswer: \"To the right of the cake, is it a man or a woman?\""
        print(f"final_answer: {[extract_answer(text48, True, 'ASSISTANT', task_name)]}") # ['To the right of the cake']
        # ['To the right of the cake, is it a man or a woman?']
        # ['To the right of the cake, is it a man or a woman']

        text49 = "The question in the image is: \"How do the thongs look, colorful or black and white?\"\n\nBased on the visual information, the answer is: The thongs look colorful."
        print(f"final_answer: {[extract_answer(text49, True, 'ASSISTANT', task_name)]}") # ['The thongs look colorful']
        # textvqa: ['The thongs look colorful.']

        text50 = "The question in the image is: \"Who is running?\"\n\nBased on the visual information, the person who is running is the one wearing black shorts and white socks."
        print(f"final_answer: {[extract_answer(text50, True, 'ASSISTANT', task_name)]}") # ['Who is running']
        # ['Who is running?']
        # ['The question in the image is: Who is running?  Based on the visual information, the person who is running is the one wearing black shorts and white socks']

        text51 = "What animal is walking on the ground? Answer the question using a single word or phrase.\n\n**Answer:** Bear"
        print(f"final_answer: {[extract_answer(text51, True, 'ASSISTANT', task_name)]}") # ['Answer']
        # ['Answer:']
        # ['Answer: Bear']

        text52 = "The building in front of the signpost is the **Gare Ferroviaire** (Train Station)."
        print(f"final_answer: {[extract_answer(text52, True, 'ASSISTANT', task_name)]}") # ['Gare Ferroviaire']
        # pope:['The building in front of the signpost is the Gare Ferroviaire (Train Station']

        text53 = "**Question:** Is there a fire hydrant on the grass?\n\n**Answer:** No"
        print(f"final_answer: {[extract_answer(text53, True, 'ASSISTANT', task_name)]}") # ['Answer']
        # ['Answer:']
        # ['Answer: No']

        text54 = "The question in the image is: \"How large are the sprinkles that are sprinkled on the cupcakes?\" The answer to this question is not provided in the image, as it is not possible to determine the size of the sprinkles from the visual information available."
        print(f"final_answer: {[extract_answer(text54, True, 'ASSISTANT', task_name)]}") # ['not provided in the image']
        # ['not provided in the image, as it is not possible to determine the size of the sprinkles from the visual information available.']
        # ['not provided in the image, as it is not possible to determine the size of the sprinkles from the visual information available']

        text55 = "The question in the image is: \"How is the clothing item that is called pink?\" The answer to this question is that the clothing item is a tennis outfit."
        print(f"final_answer: {[extract_answer(text55, True, 'ASSISTANT', task_name)]}") # ['that the clothing item is a tennis outfit']
        # textvqa: ['that the clothing item is a tennis outfit.']

        text56 = "The question in the image is: \"Is there an elephant in the snow?\" The answer is no, there is no elephant in the snow."
        print(f"final_answer: {[extract_answer(text56, True, 'ASSISTANT', task_name)]}") # ['no']
        # ['no, there is no elephant in the snow.']
        # ['no, there is no elephant in the snow']

        text57 = "Is there an elephant near the person that is wearing a coat? Answer the question using a single word or phrase. No"
        print(f"final_answer: {[extract_answer(text57, True, 'ASSISTANT', task_name)]}") # sqa: ['No']
        # ['Is there an elephant near the person that is wearing a coat? Answer the question using a single word or phrase. No']

        text58 = "Question: Are there chairs underneath the table?\nAnswer: Answer the question using a single word or phrase: No"
        print(f"final_answer: {[extract_answer(text58, True, 'ASSISTANT', task_name)]}") # ['Answer the question using a single word or phrase: No']

        ##### Samples of the original filtering code #####
        text1 = "assistant The question is: \"Is it overcast?\"\nThe answer based on visual info is: \"Yes.\""
        print(extract_answer(text1, True, 'ASSISTANT', task_name))  # "Yes"
        # textvqa: Yes.

        text2 = "Some context... The answer to the question is a small cat. Other sentences..."
        print(extract_answer(text2, True, 'ASSISTANT', task_name))  # "a small cat"
        # textvqa: a small cat. Other sentences...

        text3 = "The question is '...'.The answer is:\"plant.\"\n others..."
        print(extract_answer(text3, True, 'ASSISTANT', task_name))  # "plant"
        # plant.
        # plant. others

        text4 = "The answer is: 'Why not?!.'"
        print(extract_answer(text4, True, 'ASSISTANT', task_name))  # "Why not"
        # textvqa: Why not?!.

        text5 = "The Answer IS :  'The biggest dog in the room.' "
        print(extract_answer(text5, True, 'ASSISTANT', task_name))  # "The biggest dog in the room"
        # textvqa: The biggest dog in the room.

        text6 = "The answer is just \"No\""
        print(extract_answer(text6, True, 'ASSISTANT', task_name))  # "No"
        # pope: just No

        text7 = "The question in the image is: \"Is it overcast? Answer the question using a single word or phrase.\"\n\nThe answer based on the visual information is only : Yes."
        print(extract_answer(text7, True, 'ASSISTANT', task_name))  # "Yes"
        # textvqa: Yes.

        text8 = "The question in the image is: \"Is the freezer near the wall small or large? Answ|er the question using a single word or phrase.\"\n\nThe answer to the question is: a small cat."
        print(extract_answer(text8, True, 'ASSISTANT', task_name))  # "a small cat"
        # textvqa: a small cat.

        text9 = "The question in the image is: \"What is the person below the crowd bigger than? Answer the question using a single word or phrase.\"\n\nThe answer based on the visual information is \"plant.\""
        print(extract_answer(text9, True, 'ASSISTANT', task_name))  # "plant"
        # textvqa: plant.
