"""Base classes for numeric literal datasets."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, TextIO, cast

import pandas as pd
from pystow.utils import download, name_from_url

from ..base import LazyDataset
from ...triples import TriplesNumericLiteralsFactory
from ...typing import TorchRandomHint

__all__ = [
    "NumericPathDataset",
    "TabbedNumericDataset",
    "UnpackedRemoteNumericDataset",
    "SingleRemoteNumericDataset",
]

logger = logging.getLogger(__name__)


class NumericPathDataset(LazyDataset):
    """Contains a lazy reference to a training, testing, and validation dataset."""

    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        training_path: str | pathlib.Path | TextIO,
        testing_path: str | pathlib.Path | TextIO,
        validation_path: str | pathlib.Path | TextIO,
        literals_path: str | pathlib.Path | TextIO,
        eager: bool = False,
        create_inverse_triples: bool = False,
    ) -> None:
        """Initialize the dataset.

        :param training_path: Path to the training triples file or training triples file.
        :param testing_path: Path to the testing triples file or testing triples file.
        :param validation_path: Path to the validation triples file or validation triples file.
        :param literals_path: Path to the literals triples file or literal triples file
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        """
        self.training_path = training_path
        self.testing_path = testing_path
        self.validation_path = validation_path
        self.literals_path = literals_path

        self._create_inverse_triples = create_inverse_triples

        if eager:
            self._load()
            self._load_validation()

    def _load(self) -> None:
        self._training = self.triples_factory_cls.from_path(
            path=self.training_path,
            path_to_numeric_triples=self.literals_path,
            create_inverse_triples=self._create_inverse_triples,
        )
        self._testing = self.triples_factory_cls.from_path(
            path=self.testing_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def _load_validation(self) -> None:
        # don't call this function by itself. assumes called through the `validation`
        # property and the _training factory has already been loaded
        assert self._training is not None
        self._validation = self.triples_factory_cls.from_path(
            path=self.validation_path,
            path_to_numeric_triples=self.literals_path,
            entity_to_id=self._training.entity_to_id,  # share entity index with training
            relation_to_id=self._training.relation_to_id,  # share relation index with training
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{self.training_path}", testing_path="{self.testing_path}",'
            f' validation_path="{self.validation_path}", literals_path="{self.literals_path}")'
        )

    def _summary_rows(self):
        rv = super()._summary_rows()
        tf = self.training
        assert isinstance(tf, TriplesNumericLiteralsFactory)
        n_relations = len(tf.literals_to_id)
        n_triples = n_relations * tf.num_entities
        rv.append(("Literals", "-", n_relations, n_triples))
        return rv


class TabbedNumericDataset(LazyDataset):
    """A dataset that loads edge triples and numeric literal triples from separate DataFrames and auto-splits them."""

    ratios: ClassVar[Sequence[float]] = (0.8, 0.1, 0.1)
    triples_factory_cls = TriplesNumericLiteralsFactory

    def __init__(
        self,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
    ):
        """Initialize dataset.

        :param cache_root: An optional directory to store the extracted files. Is none is given, the default PyKEEN
            directory is used. This is defined either by the environment variable ``PYKEEN_HOME`` or defaults to
            ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split reproducible.
        """
        self.cache_root = self._help_cache(cache_root)

        self.random_state = random_state
        self._create_inverse_triples = create_inverse_triples
        self._training = None
        self._testing = None
        self._validation = None

        if eager:
            self._load()

    def _get_path(self) -> pathlib.Path | None:
        """Get the path of the data if there's a single file."""

    def _get_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def _get_numeric_df(self) -> pd.DataFrame:
        """Return the numeric literals as a DataFrame with columns (entity, attribute, value).

        :raises NotImplementedError: always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def _load(self) -> None:
        df = self._get_df()
        numeric_df = self._get_numeric_df()
        path = self._get_path()
        tf = self.triples_factory_cls.from_labeled_triples(
            triples=df.values,
            numeric_triples=numeric_df.values,
            create_inverse_triples=self._create_inverse_triples,
            metadata={"path": path} if path else None,
        )
        self._training, self._testing, self._validation = cast(
            tuple[TriplesNumericLiteralsFactory, TriplesNumericLiteralsFactory, TriplesNumericLiteralsFactory],
            tf.split(
                ratios=self.ratios,
                random_state=self.random_state,
            ),
        )

    def _load_validation(self) -> None:
        pass  # already loaded by _load()


class UnpackedRemoteNumericDataset(NumericPathDataset):
    """A numeric literal dataset with train/test/validation/literals as URLs."""

    def __init__(
        self,
        training_url: str,
        testing_url: str,
        validation_url: str,
        literals_url: str,
        cache_root: str | pathlib.Path | None = None,
        force: bool = False,
        eager: bool = False,
        create_inverse_triples: bool = False,
        download_kwargs: Mapping[str, Any] | None = None,
        extra_urls: Mapping[str, str] | None = None,
        extra_paths: Mapping[str, str | pathlib.Path] | None = None,
    ) -> None:
        """Initialize the dataset.

        :param training_url: The URL of the training triples file.
        :param testing_url: The URL of the testing triples file.
        :param validation_url: The URL of the validation triples file.
        :param literals_url: The URL of the numeric literals triples file.
        :param cache_root:
            An optional directory to cache downloaded files. If not given, use a
            subdirectory under the PyKEEN datasets directory.
        :param force: If true, redownload cached files.
        :param eager: Whether to load triples factories immediately.
        :param create_inverse_triples: Whether to create inverse training triples.
        :param download_kwargs: Keyword arguments for :func:`pystow.utils.download`.
        :param extra_urls:
            Optional mapping of extra resource names to URLs. Useful for downloading
            additional artifacts such as labels files.
        :param extra_paths:
            Optional mapping from extra resource names to local file names/paths under
            ``cache_root``. If omitted, file names are inferred from URLs.
        """
        self.cache_root = self._help_cache(cache_root)
        self.training_url = training_url
        self.testing_url = testing_url
        self.validation_url = validation_url
        self.literals_url = literals_url
        self.extra_urls = {} if extra_urls is None else dict(extra_urls)

        training_path = self.cache_root.joinpath(name_from_url(self.training_url))
        testing_path = self.cache_root.joinpath(name_from_url(self.testing_url))
        validation_path = self.cache_root.joinpath(name_from_url(self.validation_url))
        literals_path = self.cache_root.joinpath(name_from_url(self.literals_url))
        self.extra_paths: dict[str, pathlib.Path] = {
            key: self.cache_root.joinpath(pathlib.Path(path))
            for key, path in ({} if extra_paths is None else dict(extra_paths)).items()
        }

        download_kwargs = {} if download_kwargs is None else dict(download_kwargs)
        download_kwargs.setdefault("backend", "urllib")

        for url, path in [
            (self.training_url, training_path),
            (self.testing_url, testing_path),
            (self.validation_url, validation_path),
            (self.literals_url, literals_path),
        ]:
            if force or not path.is_file():
                download(url, path, **download_kwargs)

        for key, url in self.extra_urls.items():
            extra_path = self.extra_paths.get(key)
            if extra_path is None:
                extra_path = self.cache_root.joinpath(name_from_url(url))
                self.extra_paths[key] = extra_path
            if force or not extra_path.is_file():
                download(url, extra_path, **download_kwargs)

        super().__init__(
            training_path=training_path,
            testing_path=testing_path,
            validation_path=validation_path,
            literals_path=literals_path,
            eager=eager,
            create_inverse_triples=create_inverse_triples,
        )


class SingleRemoteNumericDataset(TabbedNumericDataset):
    """A numeric literal dataset backed by a single remote triples TSV and a separate remote literals TSV.

    The triples file is downloaded on first use, split automatically into training/testing/validation
    sets, and the literal matrix is built from the literals file.
    """

    def __init__(
        self,
        url: str,
        literals_url: str,
        name: str | None = None,
        literals_name: str | None = None,
        cache_root: str | None = None,
        eager: bool = False,
        create_inverse_triples: bool = False,
        random_state: TorchRandomHint = None,
        download_kwargs: dict[str, Any] | None = None,
        read_csv_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize dataset.

        :param url: The URL from which to download the main triples TSV file.
        :param literals_url: The URL from which to download the numeric literals TSV file.
            The file must be tab-separated with three columns: head entity, attribute name, numeric value.
        :param name: The local filename for the cached triples file. If not given, the filename is
            inferred from the end of ``url`` via :func:`pystow.utils.name_from_url`.
        :param literals_name: The local filename for the cached literals file. If not given, the
            filename is inferred from ``literals_url``.
        :param cache_root: An optional directory to store the downloaded files. If none is given, the
            default PyKEEN directory is used. This is defined either by the environment variable
            ``PYKEEN_HOME`` or defaults to ``~/.pykeen``.
        :param eager: Should the data be loaded eagerly? Defaults to false.
        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param random_state: An optional random state to make the training/testing/validation split
            reproducible.
        :param download_kwargs: Keyword arguments to pass through to :func:`pystow.utils.download`.
            Applied to both the triples and the literals file.
        :param read_csv_kwargs: Keyword arguments to pass through to :func:`pandas.read_csv` for the
            main triples file. Defaults to tab-separated. The literals file always uses tab-separation.

        :raises ValueError: If there is no ``url`` specified and the triples file does not already
            exist at the calculated path.
        :raises ValueError: If there is no ``literals_url`` specified and the literals file does not
            already exist at the calculated literals path.
        """
        super().__init__(
            cache_root=cache_root,
            create_inverse_triples=create_inverse_triples,
            random_state=random_state,
            eager=False,  # deferred: triggered at the bottom of this __init__
        )

        self.url = url
        self.literals_url = literals_url
        self.name = name or name_from_url(url)
        self.literals_name = literals_name or name_from_url(literals_url)

        self.download_kwargs = download_kwargs or {}
        self.read_csv_kwargs = dict(read_csv_kwargs or {})
        self.read_csv_kwargs.setdefault("sep", "\t")

        if not self._get_path().is_file() and not self.url:
            raise ValueError(f"must specify url to download from since path does not exist: {self._get_path()}")
        if not self._get_literals_path().is_file() and not self.literals_url:
            raise ValueError(
                f"must specify literals_url to download from since path does not exist: {self._get_literals_path()}"
            )

        if eager:
            self._load()

    def _get_path(self) -> pathlib.Path:
        return self.cache_root.joinpath(self.name)

    def _get_literals_path(self) -> pathlib.Path:
        return self.cache_root.joinpath(self.literals_name)

    def _get_df(self) -> pd.DataFrame:
        if not self._get_path().is_file():
            logger.info("downloading triples from %s to %s", self.url, self._get_path())
            download(url=self.url, path=self._get_path(), **self.download_kwargs)  # noqa:S310
        df = pd.read_csv(self._get_path(), **self.read_csv_kwargs)
        usecols = self.read_csv_kwargs.get("usecols")
        if usecols is not None:
            logger.info("reordering columns: %s", usecols)
            df = df[usecols]
        return df

    def _get_numeric_df(self) -> pd.DataFrame:
        if not self._get_literals_path().is_file():
            logger.info("downloading literals from %s to %s", self.literals_url, self._get_literals_path())
            download(url=self.literals_url, path=self._get_literals_path(), **self.download_kwargs)  # noqa:S310
        df = pd.read_csv(self._get_literals_path(), sep="\t", header=None)
        df[0] = df[0].astype(str)
        df[1] = df[1].astype(str)
        return df
