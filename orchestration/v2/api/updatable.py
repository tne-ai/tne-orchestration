"Code for defining and implementing updatable (mergeable, etc) classes and fields."

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

from pydantic import BaseModel
from pydantic.fields import Field


class UpdatableModel(BaseModel):
    "A data model class which may be updated by merging in another instance."
    pass


_UpdatablePrim = bool | int | float | str
_UpdatableList = List["_UpdatableValueT"]
_UpdatableDict = Dict[str, "_UpdatableValueT"]


_UpdatableValue = _UpdatablePrim | _UpdatableList | _UpdatableDict | UpdatableModel

_UpdatableModelT = TypeVar("_UpdatableModelT", bound=UpdatableModel)
_UpdatableValueT = TypeVar("_UpdatableValueT", bound=_UpdatableValue)

_Updater = Callable[[Optional[_UpdatableValueT], _UpdatableValueT], _UpdatableValueT]


def _values_are_concat_able(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> bool:
    assert patch is not None
    _basis_is_concat_able = isinstance(basis, (str, list))
    _patch_is_concat_able = isinstance(patch, (str, list))
    if basis is None:
        return _patch_is_concat_able
    assert type(basis) == type(patch)
    assert _basis_is_concat_able == _patch_is_concat_able
    return _basis_is_concat_able


def _values_are_merge_able(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> bool:
    assert patch is not None
    _basis_is_merge_able = isinstance(basis, (dict, UpdatableModel))
    _patch_is_merge_able = isinstance(patch, (dict, UpdatableModel))
    if basis is None:
        return _patch_is_merge_able
    assert type(basis) == type(patch)
    assert _basis_is_merge_able == _patch_is_merge_able
    return _basis_is_merge_able


def _non_updater_creator(existing_basis: _UpdatableValue) -> _Updater:
    assert existing_basis is not None

    def _non_updater(
        basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
    ) -> _UpdatableValueT:
        assert basis == existing_basis
        assert patch is None  # This hack violates Updater assumption.
        return basis

    return _non_updater


def _reject_updater(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> _UpdatableValueT:
    assert basis is not None
    assert patch is not None
    raise Exception("Rejected update.")


def _expect_equal_updater(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> _UpdatableValueT:
    assert basis is not None
    assert patch is not None
    if basis != patch:
        raise Exception(f"Expected equal values: {basis} != {patch}")
    return basis


def _set_once_updater(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> _UpdatableValueT:
    assert patch is not None
    if basis is not None:
        raise Exception(f"Unexpected pre-existing value: {basis}")
    return patch


def _replace_updater(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> _UpdatableValueT:
    assert patch is not None
    return patch


def _concat_updater(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> _UpdatableValueT:
    assert patch is not None
    if not _values_are_concat_able(basis, patch):
        raise Exception(f"Can't concat values: {basis} + {patch}")
    if basis is None:
        return patch
    assert hasattr(basis, "__add__")
    return cast(_UpdatableValueT, cast(Any, basis) + patch)


def _merge_helper(field_tuples):
    new_dict = {}
    for field_name, updater, basis_value, patch_value in field_tuples:
        if basis_value is None:
            if patch_value is not None:
                new_dict[field_name] = patch_value
        elif patch_value is None:
            assert basis_value is not None
            new_dict[field_name] = basis_value
        else:
            assert type(basis_value) == type(patch_value)
            new_dict[field_name] = updater(basis_value, patch_value)
    return new_dict


def _merge_updatable_dicts(
    basis: _UpdatableDict, patch: _UpdatableDict
) -> _UpdatableDict:
    all_keys = set(basis.keys()).union(patch.keys())
    assert None not in basis.values()
    assert None not in patch.values()
    field_tuples = []
    for field_name in all_keys:
        basis_value: Optional[_UpdatableValue] = basis.get(field_name)
        patch_value: Optional[_UpdatableValue] = patch.get(field_name)
        assert not (basis_value is None and patch_value is None)
        if patch_value is None:
            assert basis_value is not None
            updater = _non_updater_creator(basis_value)
        elif _values_are_concat_able(basis_value, patch_value):
            updater = _concat_updater
        elif _values_are_merge_able(basis_value, patch_value):
            updater = _merge_updater
        else:
            updater = _replace_updater
        field_tuples.append((field_name, updater, basis_value, patch_value))
    return _merge_helper(field_tuples)


def _list_updatable_field_name_and_updater_pairs(
    updatable_model: UpdatableModel,
) -> List[Tuple[str, _Updater]]:
    field_tuples = []
    for field_name, field_info in updatable_model.model_fields.items():
        json_schema_extra = field_info.json_schema_extra
        assert isinstance(json_schema_extra, dict)
        updater = cast(_Updater, json_schema_extra["updater"])
        field_tuples.append((field_name, updater))
    return field_tuples


def _merge_updatable_models(
    basis: UpdatableModel, patch: UpdatableModel
) -> UpdatableModel:
    assert type(basis) == type(patch)
    field_tuples = []
    for (
        field_name,
        updater,
    ) in _list_updatable_field_name_and_updater_pairs(basis):
        basis_value: Optional[_UpdatableValue] = getattr(basis, field_name)
        patch_value: Optional[_UpdatableValue] = getattr(patch, field_name)
        if basis_value is None and patch_value is None:
            continue
        if patch_value is None:
            assert basis_value is not None
            updater = _non_updater_creator(basis_value)
        field_tuples.append((field_name, updater, basis_value, patch_value))
    merged_dict = _merge_helper(field_tuples)
    return type(basis)(**merged_dict)


def _merge_updater(
    basis: Optional[_UpdatableValueT], patch: _UpdatableValueT
) -> _UpdatableValueT:
    assert patch is not None
    if not _values_are_merge_able(basis, patch):
        raise Exception(f"Can't merge values: {basis} + {patch}")
    if basis is None:
        return patch
    if isinstance(basis, dict):
        assert isinstance(patch, dict)
        return cast(_UpdatableValueT, _merge_updatable_dicts(basis, patch))
    else:
        assert isinstance(basis, UpdatableModel)
        assert isinstance(patch, UpdatableModel)
        return cast(_UpdatableValueT, _merge_updatable_models(basis, patch))


class UpdatePolicy:
    "An enum-like class defining a set of field update policies."

    reject = _reject_updater
    "Field updates should be rejected."

    expect_equal = _expect_equal_updater
    "Field updates should be equal to the non-None old value."

    set_once = _set_once_updater
    "Field updates should be non-None for a None old value."

    replace = _replace_updater
    "Field updates should be non-None and replace the old value."

    concat = _concat_updater
    "Field updates should be concatenated onto a possibly None old value."

    merge = _merge_updater
    "Field updates should be merged into a possibly None old value."


def UpdatableField(default: Optional[_UpdatableValue], updater: _Updater) -> Any:
    "Defines default value and UpdatePolicy for fields on UpdatableModels."

    return Field(default=default, updater=updater)  # type: ignore


def merge_updatable_models(
    basis: _UpdatableModelT, patch: _UpdatableModelT
) -> _UpdatableModelT:
    "Recursively merge `patch` into `basis` according to each field's update policy."

    return _merge_updater(basis, patch)
