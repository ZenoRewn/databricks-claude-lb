"""A table name alone is not proof of compatible idempotency storage."""
import unittest
from copy import deepcopy


def schema():
    return {'engines':{'usage_daily':'InnoDB','usage_batch_ledger':'InnoDB'},
            'ledger_columns':{'batch_id':['char','NO',36,'ascii_bin',None],
                              'event_date':['date','NO',None,None,None],
                              'payload_sha256':['char','NO',64,'ascii_bin',None],
                              'payload':['json','YES',None,None,None],
                              'created_at':['timestamp','NO',None,None,6]},
            'daily_columns':{**{n:['bigint','NO',None] for n in ('input_tokens','output_tokens','cache_creation_tokens','cache_read_tokens')},
                             **{n:['int','NO',None] for n in ('requests','errors')},'date':['date','NO',None],'model':['varchar','NO',128],
                             # Attribution column: `claude-opus-5` is spelled the
                             # same by GHCP and Foundry, so the name alone cannot
                             # separate their tariffs.
                             'provider':['varchar','NO',32]},
            'unique_indexes':{'usage_batch_ledger':{'PRIMARY':['batch_id']},'usage_daily':{'PRIMARY':['date','model','provider']}},
            'event_date_index':['event_date']}


class SchemaGateTests(unittest.TestCase):
    def test_exact_additive_schema_is_accepted(self):
        from release_probe import validate_schema
        validate_schema(schema())
    def test_wrong_width_collation_or_extra_unique_key_is_rejected(self):
        from release_probe import validate_schema
        bad=[]
        s=schema();s['ledger_columns']['batch_id'][2]=35;bad.append(s)
        s=schema();s['ledger_columns']['payload_sha256'][3]='ascii_general_ci';bad.append(s)
        s=schema();s['unique_indexes']['usage_daily']['wrong']=['date'];bad.append(s)
        s=schema();s['event_date_index']=[];bad.append(s)
        s=schema();s['daily_columns']['input_tokens'][0]='int';bad.append(s)
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_schema(value)


class LazilyCreatedPolicyTableTests(unittest.TestCase):
    """Absent is accepted, present must match exactly.

    These tables are created on first use, so requiring them would block the
    very release that introduces them. Once they exist, drift matters: a lost
    primary key would let two rows claim the same model, and which account limit
    applies would then depend on row order.
    """

    MODEL_ACCOUNT={'model_account_columns':{'model':['varchar','NO',190],
                                            'accounts':['text','NO',65535],
                                            'updated_at':['timestamp','NO',None]},
                   'model_account_unique_indexes':{'PRIMARY':['model']}}

    def test_absent_table_is_accepted(self):
        from release_probe import validate_schema
        self.assertNotIn('model_account_columns',schema())
        validate_schema(schema())

    def test_the_expected_shape_is_accepted(self):
        from release_probe import validate_schema
        validate_schema({**schema(),**deepcopy(self.MODEL_ACCOUNT)})

    def test_drift_in_the_present_table_is_rejected(self):
        from release_probe import validate_schema
        good={**schema(),**deepcopy(self.MODEL_ACCOUNT)}
        bad=[]
        s=deepcopy(good);s['model_account_columns']['model'][2]=255;bad.append(s)
        s=deepcopy(good);s['model_account_columns']['accounts'][1]='YES';bad.append(s)
        s=deepcopy(good);del s['model_account_columns']['updated_at'];bad.append(s)
        s=deepcopy(good);s['model_account_unique_indexes']={};bad.append(s)
        s=deepcopy(good);s['model_account_unique_indexes']={'PRIMARY':['accounts']};bad.append(s)
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_schema(value)


class LazilyCreatedPanelPasswordTableTests(unittest.TestCase):
    """The panel password overlay, same absent-is-accepted rule.

    This one is a single row pinned to id 1. A lost primary key would let two
    rows exist and make which verifier wins depend on row order — i.e. which
    password opens the panel. The column holds a one-way verifier, never the
    password and never recoverable ciphertext.
    """

    PANEL={'panel_password_columns':{'id':['tinyint','NO',None],
                                     'verifier':['varchar','NO',255],
                                     'updated_at':['timestamp','NO',None]},
           'panel_password_unique_indexes':{'PRIMARY':['id']}}

    def test_absent_table_is_accepted(self):
        from release_probe import validate_schema
        self.assertNotIn('panel_password_columns',schema())
        validate_schema(schema())

    def test_the_expected_shape_is_accepted(self):
        from release_probe import validate_schema
        validate_schema({**schema(),**deepcopy(self.PANEL)})

    def test_drift_in_the_present_table_is_rejected(self):
        from release_probe import validate_schema
        good={**schema(),**deepcopy(self.PANEL)}
        bad=[]
        s=deepcopy(good);s['panel_password_columns']['verifier'][2]=128;bad.append(s)
        s=deepcopy(good);s['panel_password_columns']['verifier'][1]='YES';bad.append(s)
        s=deepcopy(good);s['panel_password_columns']['id'][0]='int';bad.append(s)
        s=deepcopy(good);del s['panel_password_columns']['updated_at'];bad.append(s)
        s=deepcopy(good);s['panel_password_unique_indexes']={};bad.append(s)
        s=deepcopy(good);s['panel_password_unique_indexes']={'PRIMARY':['verifier']};bad.append(s)
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_schema(value)


class AttributionGateTests(unittest.TestCase):
    """迁移前的表必须被拒 —— 这一版的 INSERT 需要 provider 列。

    放行一个没有该列的表，等于让新代码对着不存在的列写入；放行只有
    (date, model) 主键的表，等于让两个上游的量合并进同一行，而
    `claude-opus-5` 在 GHCP 与 Foundry 完全同名，事后再也分不开。
    """

    def test_a_table_without_the_provider_column_is_refused(self):
        from release_probe import validate_schema
        s = schema()
        del s['daily_columns']['provider']
        with self.assertRaises(ValueError):
            validate_schema(s)

    def test_the_pre_migration_primary_key_is_refused(self):
        from release_probe import validate_schema
        s = schema()
        s['unique_indexes']['usage_daily']['PRIMARY'] = ['date', 'model']
        with self.assertRaises(ValueError):
            validate_schema(s)

    def test_a_widened_provider_column_is_refused(self):
        from release_probe import validate_schema
        s = schema()
        s['daily_columns']['provider'] = ['varchar', 'NO', 190]
        with self.assertRaises(ValueError):
            validate_schema(s)
