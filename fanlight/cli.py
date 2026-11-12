"""Command line interface: ``fanlight <command>`` (or ``python -m fanlight``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Settings

_COLOURS = ["\033[95m", "\033[93m", "\033[96m", "\033[92m", "\033[94m", "\033[91m"]
_RESET, _DIM, _BOLD = "\033[0m", "\033[2m", "\033[1m"


def _engine(args: argparse.Namespace):
    from .engine import Engine

    settings = Settings.from_env()
    return Engine.load(
        settings,
        demo=getattr(args, "demo", False),
        source=getattr(args, "source", "json"),
        rebuild=getattr(args, "rebuild", False),
    )


def _colour(enabled: bool):
    def paint(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if enabled else text

    return paint


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    engine = _engine(args)
    print(f"Fanlight {__version__}: {engine.index.size} stories, open http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    engine = _engine(args)
    meta = engine.meta()
    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0
    stats, ingest = meta["stats"], meta["ingest"]
    print(f"Articles kept     {ingest['kept']} of {ingest['raw_articles']}")
    print(f"  deleted         {ingest['dropped_deleted']}")
    print(f"  duplicates      {ingest['dropped_duplicate']}")
    print(f"  unpublished     {ingest['dropped_unpublished']}")
    print(f"Stories           {stats['stories']} (near-duplicate reports merged)")
    print(f"Languages         {', '.join(f'{k} {v}' for k, v in stats['languages'].items())}")
    print(f"Categories        {stats['categories']}    Tags {stats['tags']}    Outlets {stats['sources']}")
    print(f"Vocabulary        {stats['vocabulary']} terms")
    print(f"Feed clock        {meta['as_of']}")
    print(f"Imported readers  {meta['readers']['imported']} in {len(meta['cohorts'])} cohorts")
    for cohort in meta["cohorts"]:
        print(f"  {cohort['size']:>4}  {cohort['label']}")
    return 0


def cmd_readers(args: argparse.Namespace) -> int:
    engine = _engine(args)
    rows = engine.list_readers(args.limit)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for row in rows:
        follows = ", ".join(row["follows"]) or "no stated interests"
        print(
            f"{row['id']:<14} {row['kind']:<8} {'/'.join(row['languages']):<16} {row['cohort'][:34]:<36} {follows}"
        )
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    engine = _engine(args)
    profile = engine.create_reader(args.name, args.language, args.category)
    print(profile.id)
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    engine = _engine(args)
    overrides = {"diversity": args.diversity, "discovery": args.discovery}
    feed = engine.feed(args.reader, k=args.k, overrides=overrides, categories=args.category or None)
    if args.json:
        print(json.dumps(feed, indent=2, ensure_ascii=False))
        return 0
    paint = _colour(sys.stdout.isatty() and not args.no_colour)
    reader = feed["reader"]
    print(paint(f"{reader['name']}  ({reader['cohort']})", _BOLD))
    palette: dict[str, str] = {}
    for item in feed["items"]:
        category = item["categories"][0]["name"] if item["categories"] else "General"
        code = palette.setdefault(category, _COLOURS[len(palette) % len(_COLOURS)])
        marker = " *" if item["discovery"] else ""
        score = paint("[{:>3}]".format(item["match"]), code)
        print(f"{item['rank']:>3}. {score} {item['title']}{marker}")
        source = item["source"] or "Unknown outlet"
        detail = f"{category}, {source}, {item['age_hours']:.0f}h old. {'; '.join(item['reasons'])}"
        print("       " + paint(detail, _DIM))
    shape = feed["shape"]
    print(
        paint(
            f"\n{len(feed['items'])} of {feed['eligible']} eligible stories, "
            f"{len(shape['categories'])} categories, {shape['sources']} outlets, "
            f"median age {shape['median_age_hours']}h. * = discovery pick. {feed['elapsed_ms']} ms",
            _DIM,
        )
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import format_report, representation_check, simulate

    engine = _engine(args)
    report = simulate(
        engine.index, n_readers=args.readers, rounds=args.rounds, feed_size=args.k, seed=args.seed
    )
    report["representation"] = representation_check(engine.index)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(format_report(report))
    rep = report["representation"]
    if rep["agreement"] is not None:
        print(
            f"\nText index: {rep['agreement']:.0%} of each story's {rep['neighbours']} nearest "
            f"neighbours share its category (chance: {rep['chance']:.0%})."
        )
    return 0


def cmd_demo_data(args: argparse.Namespace) -> int:
    from .synth import make_raw

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = make_raw(n_articles=args.articles, n_users=args.users, seed=args.seed)
    for name, rows in raw.items():
        (out / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(raw['articles'])} synthetic articles to {out}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fanlight", description="Transparent, personalised news ranking.")
    parser.add_argument("--version", action="version", version=f"fanlight {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, func, help_text: str, data: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(func=func)
        if data:
            sub.add_argument("--demo", action="store_true", help="use the built-in synthetic newsroom")
            sub.add_argument(
                "--source", choices=["json", "mongo"], default="json", help="where to read data from"
            )
            sub.add_argument("--rebuild", action="store_true", help="ignore the index cache")
        return sub

    serve = add("serve", cmd_serve, "Run the web app and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    index = add("index", cmd_index, "Build the index and print corpus statistics")
    index.add_argument("--json", action="store_true")

    readers = add("readers", cmd_readers, "List readers")
    readers.add_argument("--limit", type=int, default=30)
    readers.add_argument("--json", action="store_true")

    create = add("create-reader", cmd_create, "Create a reader and print its id")
    create.add_argument("name")
    create.add_argument("--language", action="append", default=None, help="repeatable; default english")
    create.add_argument("--category", action="append", default=[], help="category key, repeatable")

    feed = add("feed", cmd_feed, "Print a reader's ranked feed")
    feed.add_argument("reader", help="reader id (see 'fanlight readers')")
    feed.add_argument("-k", type=int, default=20, help="number of stories")
    feed.add_argument("--category", action="append", help="only these category keys")
    feed.add_argument("--diversity", type=float, default=None)
    feed.add_argument("--discovery", type=float, default=None)
    feed.add_argument("--json", action="store_true")
    feed.add_argument("--no-colour", action="store_true")

    evaluate = add("evaluate", cmd_evaluate, "Compare ranking policies with simulated readers")
    evaluate.add_argument("--readers", type=int, default=40)
    evaluate.add_argument("--rounds", type=int, default=6)
    evaluate.add_argument("-k", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--json", action="store_true")

    demo = add("demo-data", cmd_demo_data, "Write the synthetic dataset as JSON files", data=False)
    demo.add_argument("out", help="output folder")
    demo.add_argument("--articles", type=int, default=900)
    demo.add_argument("--users", type=int, default=60)
    demo.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    if getattr(args, "command", None) == "create-reader" and args.language is None:
        args.language = ["english"]
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
