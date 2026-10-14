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
                             **{n:['int','NO',None] for n in ('requests','errors')},'date':['date','NO',None],'model':['varchar','NO',128]},
            'unique_indexes':{'usage_batch_ledger':{'PRIMARY':['batch_id']},'usage_daily':{'PRIMARY':['date','model']}},
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
