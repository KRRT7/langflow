"use strict";(self.webpackChunklangflow_docs=self.webpackChunklangflow_docs||[]).push([[1456],{5882:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>o,default:()=>h,frontMatter:()=>i,metadata:()=>a,toc:()=>c});var r=t(4848),s=t(8453);const i={title:"\u26a1\ufe0f Quickstart",sidebar_position:2,slug:"/getting-started-quickstart"},o=void 0,a={id:"Getting-Started/getting-started-quickstart",title:"\u26a1\ufe0f Quickstart",description:"Prerequisites",source:"@site/docs/Getting-Started/getting-started-quickstart.md",sourceDirName:"Getting-Started",slug:"/getting-started-quickstart",permalink:"/getting-started-quickstart",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:2,frontMatter:{title:"\u26a1\ufe0f Quickstart",sidebar_position:2,slug:"/getting-started-quickstart"},sidebar:"defaultSidebar",previous:{title:"\ud83d\udce6\xa0Installation",permalink:"/getting-started-installation"},next:{title:"\u2757\ufe0f Common Installation Issues",permalink:"/getting-started-common-installation-issues"}},l={},c=[{value:"Prerequisites",id:"b5f154a3a1d242c7bdf57acf0a552732",level:2},{value:"Hello World - Basic Prompting",id:"67e7cd59d0fa43e3926bdc75134f7472",level:2},{value:"Run the basic prompting flow",id:"27ac88f4721b42c9a9587326905b8df4",level:2},{value:"Modify the prompt for a different result",id:"5208b946024846169fe59ee206021a4f",level:2},{value:"Next steps",id:"63b6db6cb571489c86b3ae89051f1a4f",level:2}];function d(e){const n={a:"a",code:"code",h2:"h2",hr:"hr",img:"img",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.h2,{id:"b5f154a3a1d242c7bdf57acf0a552732",children:"Prerequisites"}),"\n",(0,r.jsx)(n.hr,{}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.a,{href:"https://www.python.org/downloads/release/python-3100/",children:"Python >=3.10"}),"\xa0and\xa0",(0,r.jsx)(n.a,{href:"https://pypi.org/project/pip/",children:"pip"}),"\xa0or\xa0",(0,r.jsx)(n.a,{href:"https://pipx.pypa.io/stable/installation/",children:"pipx"})]}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.a,{href:"https://platform.openai.com/",children:"OpenAI API key"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.a,{href:"/getting-started-installation",children:"Langflow installed and running"})}),"\n"]}),"\n",(0,r.jsx)(n.h2,{id:"67e7cd59d0fa43e3926bdc75134f7472",children:"Hello World - Basic Prompting"}),"\n",(0,r.jsx)(n.p,{children:"Let's start with a Prompt component to instruct an OpenAI Model."}),"\n",(0,r.jsx)(n.p,{children:"Prompts serve as the inputs to a large language model (LLM), acting as the interface between human instructions and computational tasks. By submitting natural language requests in a prompt to an LLM, you can obtain answers, generate text, and solve problems."}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["From the Langflow dashboard, click\xa0",(0,r.jsx)(n.strong,{children:"New Project"}),"."]}),"\n",(0,r.jsxs)(n.li,{children:["Select\xa0",(0,r.jsx)(n.strong,{children:"Basic Prompting"}),"."]}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.img,{src:t(939).A+"",width:"3456",height:"1756"})}),"\n",(0,r.jsxs)(n.p,{children:["This flow allows you to chat with the\xa0",(0,r.jsx)(n.strong,{children:"OpenAI"}),"\xa0model by using a\xa0",(0,r.jsx)(n.strong,{children:"Prompt"}),"\xa0to send instructions."]}),"\n",(0,r.jsxs)(n.p,{children:["Examine the\xa0",(0,r.jsx)(n.strong,{children:"Prompt"}),"\xa0component. The\xa0",(0,r.jsx)(n.strong,{children:"Template"}),"\xa0field instructs the LLM to\xa0",(0,r.jsx)(n.code,{children:"Answer the user as if you were a pirate."}),"\xa0This should be interesting..."]}),"\n",(0,r.jsxs)(n.p,{children:["To use the\xa0",(0,r.jsx)(n.strong,{children:"OpenAI"}),"\xa0component, you have two options for providing your OpenAI API Key: directly passing it to the component or creating an environment variable. For better security and manageability, creating an environment variable is recommended. Here's how to set it up:"]}),"\n",(0,r.jsxs)(n.p,{children:["In the\xa0",(0,r.jsx)(n.strong,{children:"OpenAI API Key"}),"\xa0field, click the\xa0",(0,r.jsx)(n.strong,{children:"Globe"}),"\xa0button to access environment variables, and then click\xa0",(0,r.jsx)(n.strong,{children:"Add New Variable"}),"."]}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["In the\xa0",(0,r.jsx)(n.strong,{children:"Variable Name"}),"\xa0field, enter\xa0",(0,r.jsx)(n.code,{children:"openai_api_key"}),"."]}),"\n",(0,r.jsxs)(n.li,{children:["In the\xa0",(0,r.jsx)(n.strong,{children:"Value"}),"\xa0field, paste your OpenAI API Key (",(0,r.jsx)(n.code,{children:"sk-..."}),")."]}),"\n",(0,r.jsxs)(n.li,{children:["Click\xa0",(0,r.jsx)(n.strong,{children:"Save Variable"}),"."]}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:"By creating an environment variable, you keep your API key secure and make it easier to manage across different components or projects."}),"\n",(0,r.jsx)(n.h2,{id:"27ac88f4721b42c9a9587326905b8df4",children:"Run the basic prompting flow"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["Click the\xa0",(0,r.jsx)(n.strong,{children:"Playground"}),"\xa0button. This where you can interact with your bot."]}),"\n",(0,r.jsx)(n.li,{children:"Type any message and press Enter. And... Ahoy! \ud83c\udff4\u200d\u2620\ufe0f The bot responds in a piratical manner!"}),"\n"]}),"\n",(0,r.jsx)(n.h2,{id:"5208b946024846169fe59ee206021a4f",children:"Modify the prompt for a different result"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["To modify your prompt results, in the\xa0",(0,r.jsx)(n.strong,{children:"Prompt"}),"\xa0template, click the\xa0",(0,r.jsx)(n.strong,{children:"Template"}),"\xa0field. The\xa0",(0,r.jsx)(n.strong,{children:"Edit Prompt"}),"\xa0window opens."]}),"\n",(0,r.jsxs)(n.li,{children:["Change\xa0",(0,r.jsx)(n.code,{children:"Answer the user as if you were a pirate"}),"\xa0to a different character, perhaps\xa0",(0,r.jsx)(n.code,{children:"Answer the user as if you were Harold Abelson."})]}),"\n",(0,r.jsx)(n.li,{children:"Run the basic prompting flow again. The response will be markedly different."}),"\n"]}),"\n",(0,r.jsx)(n.h2,{id:"63b6db6cb571489c86b3ae89051f1a4f",children:"Next steps"}),"\n",(0,r.jsx)(n.p,{children:"Well done! You've built your first prompt in Langflow. \ud83c\udf89"}),"\n",(0,r.jsx)(n.p,{children:"By dragging Langflow components to your workspace, you can create all sorts of interesting behaviors. Here are a couple of examples:"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.a,{href:"https://docs.langflow.org/starter-projects/memory-chatbot",children:"Memory Chatbot"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.a,{href:"https://docs.langflow.org/starter-projects/blog-writer",children:"Blog Writer"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.a,{href:"https://docs.langflow.org/starter-projects/document-qa",children:"Document QA"})}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},939:(e,n,t)=>{t.d(n,{A:()=>r});const r=t.p+"assets/images/131952085-905bd051508c94150f70756784cb94e3.png"},8453:(e,n,t)=>{t.d(n,{R:()=>o,x:()=>a});var r=t(6540);const s={},i=r.createContext(s);function o(e){const n=r.useContext(i);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:o(e.components),r.createElement(i.Provider,{value:n},e.children)}}}]);