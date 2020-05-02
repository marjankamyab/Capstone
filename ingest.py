from typing import List, TextIO, Generator, Tuple, Optional, Callable, Iterable


class Document:
    def __init__(
        self, sentences: Tuple[Tuple[str, ...], ...], labels: Tuple[Tuple[str, ...], ...]
    ) -> None:
        self.sentences: Tuple[Tuple[str, ...], ...] = tuple(sentences)
        self.labels: Tuple[Tuple[str, ...], ...] = tuple(labels)


class Corpus:
    def __init__(self, docs: Iterable[Document]) -> None:
        self.docs: Tuple[Document, ...] = tuple(docs)

    @property
    def sentences(self) -> Generator[Tuple[str, ...], None, None]:
        for doc in self.docs:
            yield from doc.sentences

    @property
    def labels(self) -> Generator[Tuple[str, ...], None, None]:
        for doc in self.docs:
            yield from doc.labels

    def __iter__(self) -> Generator[Document, None, None]:
        yield from self.docs

    def __len__(self) -> int:
        return len(self.docs)


class _CoNLLIngester:
    def __init__(
        self,
        *,
        token_func: Optional[Callable[[str], str]] = None,
        label_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.token_func = token_func
        self.label_func = label_func
        self.sentences: List[Tuple[str, ...]] = []
        self.labels: List[Tuple[str, ...]] = []

        self.sentence_tokens: List[str] = []
        self.sentence_labels: List[str] = []

    def _complete_sentence(self) -> None:
        self.sentences.append(tuple(self.sentence_tokens))
        self.labels.append(tuple(self.sentence_labels))
        self.sentence_tokens = []
        self.sentence_labels = []

    def _create_document(self) -> Document:
        assert self.sentences, "No sentences to create document with"
        assert not self.sentence_tokens, "Extra tokens remaining"
        assert not self.sentence_labels, "Extra labels remaining"
        doc = Document(tuple(self.sentences), tuple(self.labels))
        self.sentences = []
        self.labels = []
        return doc

    def extract_docs(self, corpus: TextIO) -> Generator[Document, None, None]:
        for line in corpus:
            # Check that we haven't gotten out of sync
            assert len(self.sentences) == len(self.labels)
            assert len(self.sentence_tokens) == len(self.sentence_labels)

            if line.startswith("-DOCSTART-"):
                # Add the last sentence if needed
                if self.sentence_tokens:
                    self._complete_sentence()

                # Create a document if there are sentences, otherwise we are at the first
                # docstart in the corpus and there's nothing to create
                if self.sentences:
                    yield self._create_document()

            if line.strip():
                # Sample line:
                # German JJ B-NP B-MISC
                fields = line.split()

                token = fields[0]
                if self.token_func:
                    token = self.token_func(token)
                self.sentence_tokens.append(token)

                label = fields[-1]
                if self.label_func:
                    label = self.label_func(label)
                self.sentence_labels.append(label)
            else:
                # End of sentence if there are tokens, otherwise this is a blank leading
                # line before the first sentence and there's nothing to do.
                if self.sentence_tokens:
                    self._complete_sentence()

        # Finish off document
        if self.sentence_tokens:
            self._complete_sentence()

        yield self._create_document()


def load_conll(
    path: str,
    *,
    token_func: Optional[Callable[[str], str]] = None,
    label_func: Optional[Callable[[str], str]] = None,
    encoding: str = "utf8",
) -> Corpus:
    """Load a Corpus of from the specified CoNLL-format file.

    If provided, token_func and label_func will be applied to each token or label, respectively.
    """
    with open(path, encoding=encoding) as corpus:
        return Corpus(
            _CoNLLIngester(token_func=token_func, label_func=label_func).extract_docs(
                corpus
            )
        )



