"use strict";(self.webpackChunklangflow_docs=self.webpackChunklangflow_docs||[]).push([[7338],{5056:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>c,contentTitle:()=>d,default:()=>h,frontMatter:()=>a,metadata:()=>i,toc:()=>l});var t=s(4848),r=s(8453);const a={title:"Helpers",sidebar_position:4,slug:"/components-helpers"},d=void 0,i={id:"Components/components-helpers",title:"Helpers",description:"Chat memory",source:"@site/docs/Components/components-helpers.md",sourceDirName:"Components",slug:"/components-helpers",permalink:"/components-helpers",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:4,frontMatter:{title:"Helpers",sidebar_position:4,slug:"/components-helpers"},sidebar:"defaultSidebar",previous:{title:"Data",permalink:"/components-data"},next:{title:"Models",permalink:"/components-models"}},c={},l=[{value:"Chat memory",id:"304dc4a3bea74efb9068093ff18a56ad",level:2},{value:"Parameters",id:"e0af57d97f844ce99789958161d19767",level:3},{value:"Combine text",id:"13443183e6054d0694d65f8df08833d5",level:3},{value:"Parameters",id:"246676d119604fc5bf1be85fe93044aa",level:3},{value:"Create record",id:"506f43345854473b8199631bf68a3b4a",level:3},{value:"Parameters",id:"08735e90bd10406695771bad8a95976a",level:3},{value:"Custom component",id:"cda421d4bccb4e7db2e48615884ed753",level:3},{value:"Parameters",id:"04f9eb5e6da4431593a5bee8831f2327",level:3},{value:"Documents to Data",id:"53a6a99a54f0435e9209169cf7730c55",level:3},{value:"Parameters",id:"0eb5fce528774c2db4a3677973e75cf8",level:3},{value:"ID generator",id:"4a8fbfb95ebe44ee8718725546db5393",level:3},{value:"Parameters",id:"4629dd15594c47399c97d9511060e114",level:3},{value:"Message history",id:"6a1a60688641490197c6443df573960e",level:3},{value:"Parameters",id:"31c7fc2a3e8c4f7c89f923e700f4ea34",level:3},{value:"Data to text",id:"f60ab5bbc0db4b27b427897eba97fe29",level:3},{value:"Parameters",id:"01b91376569149a49cfcfd9321323688",level:3},{value:"Split text",id:"210be0ae518d411695d6caafdd7700eb",level:3},{value:"Parameters",id:"04197fcd05e64e10b189de1171a32682",level:3},{value:"Update record",id:"d3b6116dfd8d4af080ad01bc8fd2b0b3",level:3},{value:"Parameters",id:"c830224edc1d486aaaa5e2889f4f6689",level:3}];function o(e){const n={a:"a",code:"code",h2:"h2",h3:"h3",hr:"hr",li:"li",p:"p",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h2,{id:"304dc4a3bea74efb9068093ff18a56ad",children:"Chat memory"}),"\n",(0,t.jsx)(n.p,{children:"This component retrieves stored chat messages based on a specific session ID."}),"\n",(0,t.jsx)(n.h3,{id:"e0af57d97f844ce99789958161d19767",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Sender type:"}),'\xa0Choose the sender type from options like "Machine", "User", or "Both".']}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Sender name:"}),"\xa0(Optional) The name of the sender."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Number of messages:"}),"\xa0Number of messages to retrieve."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Session ID:"}),"\xa0The session ID of the chat history."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Order:"}),'\xa0Choose the message order, either "Ascending" or "Descending".']}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Data template:"}),"\xa0(Optional) Template to convert a record to text. If left empty, the system dynamically sets it to the record's text key."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"13443183e6054d0694d65f8df08833d5",children:"Combine text"}),"\n",(0,t.jsx)(n.p,{children:"This component concatenates two text sources into a single text chunk using a specified delimiter."}),"\n",(0,t.jsx)(n.h3,{id:"246676d119604fc5bf1be85fe93044aa",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"First text:"}),"\xa0The first text input to concatenate."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Second text:"}),"\xa0The second text input to concatenate."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Delimiter:"}),"\xa0A string used to separate the two text inputs. Defaults to a space."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"506f43345854473b8199631bf68a3b4a",children:"Create record"}),"\n",(0,t.jsx)(n.p,{children:"This component dynamically creates a record with a specified number of fields."}),"\n",(0,t.jsx)(n.h3,{id:"08735e90bd10406695771bad8a95976a",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Number of fields:"}),"\xa0Number of fields to be added to the record."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Text key:"}),"\xa0Key used as text."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"cda421d4bccb4e7db2e48615884ed753",children:"Custom component"}),"\n",(0,t.jsx)(n.p,{children:"Use this component as a template to create your custom component."}),"\n",(0,t.jsx)(n.h3,{id:"04f9eb5e6da4431593a5bee8831f2327",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Parameter:"}),"\xa0Describe the purpose of this parameter."]}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"INFO"}),"\n",(0,t.jsxs)(n.p,{children:["Customize the\xa0",(0,t.jsx)(n.code,{children:"build_config"}),"\xa0and\xa0",(0,t.jsx)(n.code,{children:"build"}),"\xa0methods according to your requirements."]}),"\n",(0,t.jsxs)(n.p,{children:["Learn more about creating custom components at\xa0",(0,t.jsx)(n.a,{href:"http://docs.langflow.org/components/custom",children:"Custom Component"}),"."]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"53a6a99a54f0435e9209169cf7730c55",children:"Documents to Data"}),"\n",(0,t.jsx)(n.p,{children:"Convert LangChain documents into Data."}),"\n",(0,t.jsx)(n.h3,{id:"0eb5fce528774c2db4a3677973e75cf8",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Documents:"}),"\xa0Documents to be converted into Data."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"4a8fbfb95ebe44ee8718725546db5393",children:"ID generator"}),"\n",(0,t.jsx)(n.p,{children:"Generates a unique ID."}),"\n",(0,t.jsx)(n.h3,{id:"4629dd15594c47399c97d9511060e114",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Value:"}),"\xa0Unique ID generated."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"6a1a60688641490197c6443df573960e",children:"Message history"}),"\n",(0,t.jsx)(n.p,{children:"Retrieves stored chat messages based on a specific session ID."}),"\n",(0,t.jsx)(n.h3,{id:"31c7fc2a3e8c4f7c89f923e700f4ea34",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Sender type:"}),"\xa0Options for the sender type."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Sender name:"}),"\xa0Sender name."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Number of messages:"}),"\xa0Number of messages to retrieve."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Session ID:"}),"\xa0Session ID of the chat history."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Order:"}),"\xa0Order of the messages."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"f60ab5bbc0db4b27b427897eba97fe29",children:"Data to text"}),"\n",(0,t.jsx)(n.p,{children:"Convert Data into plain text following a specified template."}),"\n",(0,t.jsx)(n.h3,{id:"01b91376569149a49cfcfd9321323688",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Data:"}),"\xa0The Data to convert to text."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Template:"}),"\xa0The template used for formatting the Data. It can contain keys like\xa0",(0,t.jsx)(n.code,{children:"{text}"}),",\xa0",(0,t.jsx)(n.code,{children:"{data}"}),", or any other key in the record."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"210be0ae518d411695d6caafdd7700eb",children:"Split text"}),"\n",(0,t.jsx)(n.p,{children:"Split text into chunks of a specified length."}),"\n",(0,t.jsx)(n.h3,{id:"04197fcd05e64e10b189de1171a32682",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Texts:"}),"\xa0Texts to split."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Separators:"}),"\xa0Characters to split on. Defaults to a space."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Max chunk size:"}),"\xa0The maximum length (in characters) of each chunk."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Chunk overlap:"}),"\xa0The amount of character overlap between chunks."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Recursive:"}),"\xa0Whether to split recursively."]}),"\n"]}),"\n",(0,t.jsx)(n.hr,{}),"\n",(0,t.jsx)(n.h3,{id:"d3b6116dfd8d4af080ad01bc8fd2b0b3",children:"Update record"}),"\n",(0,t.jsx)(n.p,{children:"Update a record with text-based key/value pairs, similar to updating a Python dictionary."}),"\n",(0,t.jsx)(n.h3,{id:"c830224edc1d486aaaa5e2889f4f6689",children:"Parameters"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Data:"}),"\xa0The record to update."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"New data:"}),"\xa0The new data to update the record with."]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(o,{...e})}):o(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>d,x:()=>i});var t=s(6540);const r={},a=t.createContext(r);function d(e){const n=t.useContext(a);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function i(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:d(e.components),t.createElement(a.Provider,{value:n},e.children)}}}]);