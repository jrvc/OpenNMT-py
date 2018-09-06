#!/usr/bin/env python

# read in a log from multi-encoder training, report perplexities for each 
# language over time in CSV format

import sys
import re
from collections import defaultdict


def main(field='perplexity'):
    # perplexities grouped by step and language pair
    ppls = defaultdict(lambda: defaultdict(float))

    # which training step we're currently in
    step = None

    # which language pair are we evaluating
    lang_pair = None

    for line in sys.stdin:
        # set new step number
        step_match = re.search("Step (\d+)", line)
        if step_match:
            step_ = int(step_match.group(1))
            if not step or step_ > step:
                step = step_  # hack to avoid matching validation-run steps

        # update current language pair
        lang_pair_match = re.search("language pair: .'(\w+)', '(\w+)'", line)
        if lang_pair_match:
            lang_pair = '-'.join([
                lang_pair_match.group(1),
                lang_pair_match.group(2)])

        # report perplexity for language pair
        ppl_match = re.search("Validation perplexity: (.*)", line)
        if ppl_match:
            ppl = float(ppl_match.group(1))
            ppls[step][lang_pair] = ppl

    # all language pairs
    langs = sorted(list(ppls.values())[0])

    # print CSV header
    print(','.join(["step"] + langs))

    # for each step, print line with perplexities for all language pairs
    for s in sorted(ppls.keys()):
        step_ppls = [ppls[s][lang] for lang in langs]
        print(",".join([str(s)] + [str(p) for p in step_ppls]))


if __name__ == "__main__":
    main()
