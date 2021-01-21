"""
Error annotation for Chinese sentences
"""
import argparse
from collections import namedtuple
from itertools import groupby
import numpy as np
from tqdm import tqdm

Annotation = namedtuple(
    "Annotation",
    [
        "op",
        "toks",
        "inds",
    ],
)


class ZhAnnotator:
    def __init__(self, tokenizer, semantic_dict, semantic_classes, annotator_id=0):
        self.tokenizer = tokenizer
        self.semantic_dict = semantic_dict
        self.semantic_classes = semantic_classes
        self.annotator_id = annotator_id

    @classmethod
    def create_default(cls, annotator_id=0):
        """
        Default parameters used in the paper
        """
        semantic_dict, semantic_classes = cls.read_cilin()
        tokenizer = ZhTokenizer()
        tokenizer.method("char")
        return cls(tokenizer, semantic_dict, semantic_classes, annotator_id)

    def __call__(self, src: str, tgt: str):
        """
        Align sentences and annotate them with error type information
        """
        align = ZhAlignment(
            src,
            tgt,
            tokenizer=self.tokenizer,
            semantic_dict=self.semantic_dict,
            semantic_classes=self.semantic_classes,
        )
        merge = ZhMerger(align)

        annotations = []
        src_seg = [x for x, _ in align.src_seg]
        tgt_seg = [x for x, _ in align.tgt_seg]
        for edit in merge.edits:
            op = edit[0][0]
            # src_tok = ' '.join(src_seg[edit[1]:edit[2]])
            tgt_tok = " ".join(tgt_seg[edit[3] : edit[4]])

            # convert our alignment ops into edit ops
            # S -> S (ubsitute)
            # D -> R (emove)
            if op == "D":
                op = "R"
            # I -> M (issing)
            elif op == "I":
                op = "M"
            # T -> W (ord Order)
            elif op == "T":
                op = "W"

            if op == "R":
                annotations.append(Annotation(op, "-NONE-", (edit[1], edit[2])))
            elif op == "M":
                annotations.append(Annotation(op, tgt_tok, (edit[1], edit[2])))
            elif op == "S":
                annotations.append(Annotation(op, tgt_tok, (edit[1], edit[2])))
            elif op == "W":
                annotations.append(Annotation(op, tgt_tok, (edit[1], edit[2])))

        # convert to text form
        annotations_out = ["S " + " ".join(src_seg) + "\n"]
        for annotation in annotations:
            op, tok, inds = annotation
            a_str = f"A {inds[0]} {inds[1]}|||{op}|||{tok}|||REQUIRED|||-NONE-|||{self.annotator_id}\n"
            annotations_out.append(a_str)
        annotations_out.append("\n")

        return annotations_out

    @staticmethod
    def read_cilin():
        """
        Cilin 詞林 is a thesaurus with semantic information
        """
        # TODO -- fix this path
        lines = open("data/cilin.txt", "r", encoding="gbk").read().strip().split("\n")
        semantic_dict = {}
        semantic_classes = {}
        for line in lines:
            code, *words = line.split(" ")
            for word in words:
                semantic_dict[word] = code
            # make reverse dict
            if code in semantic_classes:
                semantic_classes[code] += words
            else:
                semantic_classes[code] = words
        return semantic_dict, semantic_classes


class ZhTokenizer:
    """
    NOTE: Originally I experimented with many different tokenizers, but found the segmentation to be poor quality
    as tokenizers are trained to segment grammatically correct sentences, therefore I chose to simply use character
    level tokenization. A potential future work would be to extend this.
    """

    def __init__(self):
        self.tokenizer = None

    def method(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_str):

        if self.tokenizer == "char":
            input_str = "".join(input_str.split(" "))
            input_seg = [(char, "none") for char in input_str]
        else:
            raise NotImplementedError

        return input_seg


class ZhMerger:
    """
    Merge certain operations from aligned sentences
    """

    def __init__(self, align_obj):
        self.alignment = align_obj
        self.edits = self.merge()

    def merge(self):
        """
        Based on ERRANT's merge, adapted for Chinese
        """

        def merge_edits(seq, tag="X"):
            if seq:
                return [(tag, seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
            else:
                return seq

        def process_seq(seq):
            if len(seq) <= 1:
                return seq

            ops = [op[0] for op in seq]
            if set(ops) == {"D"} or set(ops) == {"I"}:
                return merge_edits(seq, set(ops).pop())

            if set(ops) == {"D", "I"} or set(ops) == {"I", "D"}:
                # do not merge this pattern
                return seq

            if set(ops) == {"S"}:
                return seq

            return merge_edits(seq, "S")

        edits = []
        # Split alignment into groups of M, T and rest. (T has a number after it)
        for op, group in groupby(
            self.alignment.align_seq,
            lambda x: x[0][0] if x[0][0] in {"M", "T"} else False,
        ):
            group = list(group)
            if op == "M":
                for seq in group:
                    edits.append(seq)
            # T is always split TODO: Evaluate this
            elif op == "T":
                for seq in group:
                    edits.append(seq)
            # Process D, I and S subsequence
            else:
                # Turn the processed sequence into edits
                processed = process_seq(group)
                for seq in processed:
                    edits.append(seq)
        # Find "I M D" or "D M I" patterns
        # Ex:
        # M     I   M   D   M
        # 我        決   了  心
        # 我    下  決       心
        filtered_edits = []
        i = 0
        while i < len(edits):
            e1 = edits[i][0][0]

            if i < len(edits) - 2:
                e2 = edits[i + 1][0][0]
                e3 = edits[i + 2][0][0]

                if (e1 == "I" and e2 == "M" and e3 == "D") or (
                    e1 == "D" and e2 == "M" and e3 == "I"
                ):

                    group = [edits[i], edits[i + 1], edits[i + 2]]
                    processed = merge_edits(group, "S")
                    for seq in processed:
                        filtered_edits.append(seq)
                    i += 3

                else:
                    if e1 != "M":
                        filtered_edits.append(edits[i])
                    i += 1
            else:
                if e1 != "M":
                    filtered_edits.append(edits[i])
                i += 1
        # In rare cases with word-level tokenization, the following error can occur:
        # M     D   S       M
        # 有    時  住      上層
        # 有        時住    上層
        # Which results in S: 時住 --> 時住
        # We need to filter this case out
        second_filter = []
        src = [x for x, _ in self.alignment.src_seg]
        tgt = [x for x, _ in self.alignment.tgt_seg]
        for edit in filtered_edits:
            tok1 = "".join(src[edit[1] : edit[2]])
            tok2 = "".join(tgt[edit[3] : edit[4]])
            if tok1 != tok2:
                second_filter.append(edit)
        return second_filter

    def display(self):
        for edit in self.edits:
            op = edit[0]
            src = [x for x, _ in self.alignment.src_seg]
            tgt = [x for x, _ in self.alignment.tgt_seg]
            src = " ".join(src[edit[1] : edit[2]])
            tgt = " ".join(tgt[edit[3] : edit[4]])
            print(f"{op}:\t{src}\t-->\t{tgt}")


class ZhAlignment:
    def __init__(
        self,
        src: str,
        tgt: str,
        tokenizer: ZhTokenizer,
        semantic_dict,
        semantic_classes,
        verbose: bool = False,
    ):
        self.insertion_cost = 1
        self.deletion_cost = 1
        self.semantic_dict = semantic_dict
        self.semantic_classes = semantic_classes

        self.src = src
        self.tgt = tgt
        self.src_seg = tokenizer(src)
        self.tgt_seg = tokenizer(tgt)

        # Because we use character level tokenization, this doesn't currently use POS
        self._open_pos = {}

        if verbose:
            print("========== Seg. and POS: ==========")
            print(self.src_seg)
            print(self.tgt_seg)

        self.cost_matrix, self.oper_matrix = self.align()

        if verbose:
            print("========== Cost Matrix ==========")
            print(self.cost_matrix)
            print("========== Oper Matrix ==========")
            print(self.oper_matrix)

        self.align_seq = self.get_cheapest_align_seq()

        if verbose:
            print("========== Alignment ==========")
            print(self.align_seq)

        if verbose:
            print("========== Results ==========")
            for a in self.align_seq:
                print(a[0], self.src_seg[a[1] : a[2]], self.tgt_seg[a[3] : a[4]])

    def _get_semantic_class(self, word):
        """
        NOTE: Based on the paper:
        Improved-Edit-Distance Kernel for Chinese Relation Extraction
        """
        if word in self.semantic_dict:
            code = self.semantic_dict[word]
            high, mid, low = code[0], code[1], code[2:4]
            return high, mid, low
        else:  # unknown
            return None

    @staticmethod
    def _get_class_diff(a_class, b_class):
        """
        d == 3 for equivalent semantics
        d == 0 for completely different semantics
        """
        d = sum([a == b for a, b in zip(a_class, b_class)])
        return d

    def _get_semantic_cost(self, a, b):
        a_class = self._get_semantic_class(a)
        b_class = self._get_semantic_class(b)
        # unknown class, default to 1
        if a_class is None or b_class is None:
            return 4
        elif a_class == b_class:
            return 0
        else:
            return 2 * (3 - self._get_class_diff(a_class, b_class))

    def _get_pos_cost(self, a_pos, b_pos):
        if a_pos == b_pos:
            return 0
        elif a_pos in self._open_pos and b_pos in self._open_pos:
            return 0.25
        else:
            return 0.5

    @staticmethod
    def _get_char_cost(a, b):
        """
        NOTE: This is a replacement of ERRANTS lemma cost for Chinese
        """
        if a == b:
            return 0
        elif any([a_ in b for a_ in a]):
            # Contains some of the same characters
            return 0.25
        else:
            return 0.5

    def get_sub_cost(self, a_seg, b_seg):
        """
        Calculate the substitution cost between words a and b
        """
        if a_seg[0] == b_seg[0]:
            return 0

        semantic_cost = self._get_semantic_cost(a_seg[0], b_seg[0]) / 6.0
        pos_cost = self._get_pos_cost(a_seg[1], b_seg[1])
        char_cost = self._get_char_cost(a_seg[0], b_seg[0])

        return semantic_cost + pos_cost + char_cost

    def align(self):
        """
        Based on ERRANT's alignment
        """
        cost_matrix = np.zeros((len(self.src_seg) + 1, len(self.tgt_seg) + 1))
        oper_matrix = np.full(
            (len(self.src_seg) + 1, len(self.tgt_seg) + 1), "O", dtype=object
        )
        # Fill in the edges
        for i in range(1, len(self.src_seg) + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            oper_matrix[i][0] = "D"
        for j in range(1, len(self.tgt_seg) + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            oper_matrix[0][j] = "I"

        # Loop through the cost matrix
        for i in range(len(self.src_seg)):
            for j in range(len(self.tgt_seg)):
                # Matches
                if self.src_seg[i][0] == self.tgt_seg[j][0]:
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    oper_matrix[i + 1][j + 1] = "M"
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j + 1] + self.deletion_cost
                    ins_cost = cost_matrix[i + 1][j] + self.insertion_cost
                    sub_cost = cost_matrix[i][j] + self.get_sub_cost(
                        self.src_seg[i], self.tgt_seg[j]
                    )
                    # Calculate transposition cost
                    trans_cost = float("inf")
                    k = 1
                    while (
                        i - k >= 0
                        and j - k >= 0
                        and cost_matrix[i - k + 1][j - k + 1]
                        != cost_matrix[i - k][j - k]
                    ):
                        p1 = sorted([a[0] for a in self.src_seg][i - k : i + 1])
                        p2 = sorted([b[0] for b in self.tgt_seg][j - k : j + 1])
                        if p1 == p2:
                            trans_cost = cost_matrix[i - k][j - k] + k
                            break
                        k += 1

                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    ind = costs.index(min(costs))
                    cost_matrix[i + 1][j + 1] = costs[ind]

                    if ind == 0:
                        oper_matrix[i + 1][j + 1] = "T" + str(k + 1)
                    elif ind == 1:
                        oper_matrix[i + 1][j + 1] = "S"
                    elif ind == 2:
                        oper_matrix[i + 1][j + 1] = "I"
                    else:
                        oper_matrix[i + 1][j + 1] = "D"

        return cost_matrix, oper_matrix

    def get_cheapest_align_seq(self):
        i = self.oper_matrix.shape[0] - 1
        j = self.oper_matrix.shape[1] - 1
        align_seq = []
        while i + j != 0:
            op = self.oper_matrix[i][j]
            if op in {"M", "S"}:
                align_seq.append((op, i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif op == "D":
                align_seq.append((op, i - 1, i, j, j))
                i -= 1
            elif op == "I":
                align_seq.append((op, i, i, j - 1, j))
                j -= 1
            else:
                k = int(op[1:])
                align_seq.append((op, i - k, i, j - k, j))
                i -= k
                j -= k
        align_seq.reverse()
        return align_seq

    def display(self, max_len=120):
        """
        Display the alignments in the terminal in a "pretty" way
        """
        seq1 = []
        seq2 = []
        opseq = []

        for obj in self.align_seq:
            op, s11, s12, s21, s22 = obj

            src = self.src_seg[s11:s12]
            src = " ".join([x for x, _ in src])

            tgt = self.tgt_seg[s21:s22]
            tgt = " ".join([x for x, _ in tgt])

            opseq.append(op)
            seq1.append(src)
            seq2.append(tgt)

        # full width versions (SPACE is non-contiguous with ! through ~)
        SPACE = "\N{IDEOGRAPHIC SPACE}"
        EXCLA = "\N{FULLWIDTH EXCLAMATION MARK}"
        TILDE = "\N{FULLWIDTH TILDE}"
        # LEFTQUOTE = '\N{LEFT DOUBLE QUOTATION MARK}'
        # RIGHTQUOTE = '\N{RIGHT DOUBLE QUOTATION MARK}'
        # strings of ASCII and full-width characters (same order)
        west = "".join(chr(i) for i in range(ord(" "), ord("~")))
        east = SPACE + "".join(chr(i) for i in range(ord(EXCLA), ord(TILDE)))
        # deal with “ ”
        west += "“"
        west += "”"
        east += "\N{FULLWIDTH QUOTATION MARK}"
        east += "\N{FULLWIDTH QUOTATION MARK}"

        # build the translation table
        full = str.maketrans(west, east)

        opseqout = "{:8}".format(" ")
        for x in opseq:
            if x == "":
                x = SPACE
            opseqout += "{:4}".format(x).translate(full)

        seq1out = "{:8}".format("src:")
        for x in seq1:
            if x == "":
                x = SPACE
            seq1out += "{:4}".format(x).translate(full)

        seq2out = "{:8}".format("tgt:")
        for x in seq2:
            if x == "":
                x = SPACE
            seq2out += "{:4}".format(x).translate(full)

        for x in range(0, len(seq1out), max_len - 1):
            print(
                f"{opseqout[x:x+max_len-1]}\n{seq1out[x:x+max_len-1]}\n{seq2out[x:x+max_len-1]}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose input pair file to annotate")
    parser.add_argument("-s", "--src", type=str, required=True, help="Input src file")
    parser.add_argument(
        "-t", "--tgt", type=str, required=True, help="Input tgt (gold) file"
    )
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument(
        "-a", "--annotator", type=int, help="Annotator ID", required=False, default=0
    )
    args = parser.parse_args()
    print(args)

    source_lines = open(args.src, "r").read().strip().split("\n")
    target_lines = open(args.tgt, "r").read().strip().split("\n")

    annotator = ZhAnnotator.create_default(annotator_id=0)

    with open(args.output, "w") as f:
        for src, tgt in tqdm(zip(source_lines, target_lines)):
            annotations = annotator(src, tgt)
            for line in annotations:
                f.write(line)
