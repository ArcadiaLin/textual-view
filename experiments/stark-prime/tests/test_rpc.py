"""Exercise the RPC lifecycle with an early agent_end followed by native retry."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run import run_rpc


class RpcTests(unittest.TestCase):
    def test_waits_for_settled_and_uses_last_authoritative_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'work').mkdir()
            (root / 'agent').mkdir()
            fake = root / 'pi'
            fake.write_text('''#!/usr/bin/python3
import json,sys
def emit(obj):
 print(json.dumps(obj),flush=True)
for line in sys.stdin:
 command=json.loads(line)
 if command['type']=='get_state':
  emit({'id':'state','type':'response','success':True,'data':{}})
 elif command['type']=='prompt':
  emit({'id':'task','type':'response','success':True,'data':{'disposition':'started'}})
  emit({'type':'agent_end','willRetry':True})
  emit({'type':'auto_retry_start'})
  emit({'type':'message_update','assistantMessageEvent':{'type':'text_delta','delta':'ignore this'}})
  emit({'type':'message_end','message':{'role':'assistant','stopReason':'stop','content':[{'type':'text','text':'{"node_ids": [42]}'}],'usage':{'input':10,'output':5}}})
  emit({'type':'agent_end','willRetry':False})
  emit({'type':'agent_settled','aborted':False})
''')
            fake.chmod(0o755)
            with patch.dict(os.environ, {'PATH': str(root) + ':' + os.environ['PATH']}):
                final, state = asyncio.run(run_rpc(root, 'query', timeout=10, max_turns=5))
            self.assertEqual(json.loads(final), {'node_ids': [42]})
            self.assertEqual(state['event_counts']['agent_end'], 2)
            self.assertEqual(state['event_counts']['agent_settled'], 1)


if __name__ == '__main__':
    unittest.main()
