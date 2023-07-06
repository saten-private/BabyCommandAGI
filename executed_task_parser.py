from collections import deque

class ExecutedTaskParser:

    def encode(self, input_data: deque) -> str:
        output = ""
        for item in input_data:
            output += f"{item['type']}: {item['target']}\n"
            output += "```\n"
            output += f"{item['result']}\n"
            output += "```\n"
        return output
