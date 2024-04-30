from enum import Enum
from inspect import signature

import typer
from typing_extensions import Annotated


class TyperEnum(str, Enum):
    def __new__(cls, *args):
        name_and_value, extra_value = args
        obj = str.__new__(cls, name_and_value)
        obj._name_ = name_and_value
        obj._value_ = name_and_value
        obj._extra_value_ = extra_value
        return obj

    def __init__(self, name_and_value, extra_value):
        assert self._name_ == name_and_value
        assert self._value_ == name_and_value
        assert self._extra_value_ == extra_value

    def __str__(self):
        return self.value

    @property
    def extra_value(self):
        return self._extra_value_


def typer_cli(name, help):
    cli = typer.Typer(
        name=name,
        help=help,
        no_args_is_help=True,
        pretty_exceptions_enable=False,
    )
    return cli


def _command_has_non_option_parameters(command):
    sig = signature(command)
    return any(
        not any(
            isinstance(arg, typer.models.OptionInfo)
            for arg in parameter.annotation.__metadata__
        )
        for parameter in sig.parameters.values()
    )


def typer_cli_with_commands(name, help, commands):
    cli = typer_cli(name, help)
    for command in commands:
        no_args_is_help = _command_has_non_option_parameters(command)
        cli.command(no_args_is_help=no_args_is_help)(command)
    return cli


def typer_cli_with_sub_clis(name, help, sub_clis):
    cli = typer_cli(name, help)
    for sub_cli in sub_clis:
        cli.add_typer(sub_cli)
    return cli


def typer_arg(type, help):
    return Annotated[type, typer.Argument(help=help)]


def typer_opt(type, help):
    # TODO: Figure out why passing default into typer.Option causes exception.
    return Annotated[type, typer.Option(help=help)]
