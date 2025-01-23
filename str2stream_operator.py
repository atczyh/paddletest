"""The stream operator."""
import logging
import json
from typing import AsyncIterator
import asyncio
from dbgpt.core.awel import StreamifyAbsOperator
import traceback



from dbgpt.core.awel.flow import (
    IOField,
    OperatorCategory,
    ViewMetadata,
)

from dbgpt.util.i18n_utils import _




class str2streamOperator(StreamifyAbsOperator[str, dict]):
    metadata = ViewMetadata(
        label=_("str2streamtest"),
        name="str_2_stream_operator",
        category=OperatorCategory.COMMON,
        description=_("json conver stream"),
        parameters=[],
        inputs=[
            IOField.build_from(
                _("Non-Streaming String Input"),
                "not_stream_input",
                str,
                description=_("The non-streaming input."),
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Streaming Dict String Output"),
                "dict",
                dict,
                description=_("The streaming output."),
            ),
        ],
    )
    async def streamify(self, data: str) -> AsyncIterator[dict]:
        try:
            print('streamify',data)
            parsed_data=data.replace("data:```vis-db-chart",'').replace("```vis-db-chart\n",'').replace("\n```",'')
            print('streamify2',parsed_data)
            parsed_data = json.loads(parsed_data)
            data_dd=parsed_data['data']
            print(len(data_dd))
            for row in data_dd:
                yield row
        except Exception as e:
            print(f"Error processing item: {e}")
            traceback.print_exc()



