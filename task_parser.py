from typing import Dict, List
from collections import deque

class TaskParser:

    def test(self):
        test_input = """type: command
path: /workspace/
```bash
sudo apt-get update
sudo apt-get install git
git clone https://github.com/flutter/flutter.git
```

type: command
path: /workspace/
```bash
cd flutter
./bin/flutter doctor
export PATH="$PATH:`pwd`/bin"
```

type: command
path: /workspace/
```bash
cd /workspace
flutter create my_flutter_app
```

type: command
path: /workspace/
```bash
flutter channel beta
flutter upgrade
flutter config --enable-web
```

type: command
path: /workspace/
```bash
cd my_flutter_app
flutter run -d web-server --web-port=8080
```

type: write
path: /app/Dockerfile
```
FROM ubuntu:latest

RUN apt-get update && apt-get install -y git

COPY . /app

WORKDIR /app

RUN git clone https://github.com/flutter/flutter.git

ENV PATH="/workspace/flutter/bin:${PATH}"

RUN flutter doctor

RUN cd /workspace && flutter create my_flutter_app

RUN flutter channel beta && flutter upgrade && flutter config --enable-web

EXPOSE 8080

CMD ["flutter", "run", "-d", "web-server", "--web-port=8080"]
```

type: plan
```
Configure the container to expose port 8080 to the host machine and access the app from a browser outside the container.
```

type: write
path: /workspace/othello_app/lib/main.dart
```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Othello',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'Othello'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Text('Othello Game'),
      ),
    );
  }
}
```"""
#        test_input = """Example of tasks output:
#1. Start a web server: `flutter run -d web-server --web-port 8080`
#2. Make the web server accessible from outside the container: Configure the container to expose port 8080 and map it to the host machine."""
        parsed_data = self.decode(test_input)
        print(parsed_data)
        print(self.encode(parsed_data))

    def decode(self, input_string: str) -> deque:
        data = input_string.strip().split("type:")[1:]
        if len(data) == 0:
            raise ValueError("No valid items found")
        parsed_data = deque([])

        for item in data:
            item = "type:" + item.strip()
            type, path, content = self._split_data(item)
            dict = {}
            dict["type"] = type
            if path is not None:
                dict["path"] = path
            dict["content"] = content
            parsed_data.append(dict)

        if len(parsed_data) == 0:
            raise ValueError("No valid items found")

        return parsed_data
    
    def _split_data(self, input_data):
        lines = input_data.split('\n')
        type_line, path_line = None, None
        content_lines = []
        record_content = False
        has_content = False

        for line in lines:
            #line = line.strip()
            if line.startswith("type:"):
                if type_line is not None:
                    raise ValueError("Multiple type lines found")
                type_line = line[5:].strip()
            elif line.startswith("path:"):
                if path_line is not None:
                    raise ValueError("Multiple path lines found")
                path_line = line[5:].strip()
            elif line.startswith("```"):
                has_content = True
                record_content = not record_content
            elif record_content:
                content_lines.append(line)

        if type_line is None:
            raise ValueError("No type line found")
        if len(content_lines) == 0 and has_content == False:
            raise ValueError("No content found")

        content = "\n".join(content_lines)

        return type_line, path_line, content
    
    def encode(self, input_data: deque) -> str:
        output = ""
        for item in input_data:
            output += f"type: {item['type']}\n"
            if 'path' in item:
                output += f"path: {item['path']}\n"
            output += "```\n"
            output += f"{item['content']}\n"
            output += "```\n"
        return output
    
    def close_open_backticks(self, string: str) -> str:
        if string.count('\n```') % 2 != 0:
            string += '\n```'
        return string