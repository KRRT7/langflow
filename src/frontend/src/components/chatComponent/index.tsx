import { useContext, useEffect, useRef, useState } from "react";
import { useNodes } from "reactflow";
import { ChatType } from "../../types/chat";
import BuildTrigger from "./buildTrigger";
import ChatTrigger from "./chatTrigger";

import * as _ from "lodash";
import { TabsContext } from "../../contexts/tabsContext";
import { getBuildStatus } from "../../controllers/API";
import FormModal from "../../modals/formModal";
import { NodeType } from "../../types/flow";
import { flowManagerContext } from "../../contexts/flowManagerContext";
import IOView from "../IOview";

export default function Chat({ flow }: ChatType): JSX.Element {
  const [open, setOpen] = useState(false);
  const [canOpen, setCanOpen] = useState(false);
  const { tabsState, isBuilt, setIsBuilt } = useContext(TabsContext);
  const {checkInputandOutput,getInputTypes,getOutputTypes} = useContext(flowManagerContext)
  const [validIO, setValidIO] = useState(false);
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        (event.key === "K" || event.key === "k") &&
        (event.metaKey || event.ctrlKey) &&
        isBuilt
      ) {
        event.preventDefault();
        setOpen((oldState) => !oldState);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isBuilt]);

  useEffect(() => {
    // Define an async function within the useEffect hook
    const fetchBuildStatus = async () => {
      const response = await getBuildStatus(flow.id).catch((err) => {console.error(err)});
      setIsBuilt(!!response.built);
    };

    // Call the async function
    fetchBuildStatus();
  }, [flow]);

  const prevNodesRef = useRef<any[] | undefined>();
  const nodes: NodeType[] = useNodes();
  useEffect(() => {
    const prevNodes = prevNodesRef.current;
    const currentNodes = nodes.map((node: NodeType) =>
      _.cloneDeep(node.data.node?.template)
    );
    if (
      tabsState &&
      tabsState[flow.id] &&
      tabsState[flow.id].isPending &&
      JSON.stringify(prevNodes) !== JSON.stringify(currentNodes)
    ) {
      setIsBuilt(false);
    }
    if (
      tabsState &&
      tabsState[flow.id] &&
      tabsState[flow.id].formKeysData &&
      tabsState[flow.id].formKeysData.input_keys !== null
    ) {
      setCanOpen(true);
    } else {
      console.log(tabsState)
      setCanOpen(false);
    }

    prevNodesRef.current = currentNodes;
  }, [tabsState, flow.id]);

  useEffect((  ) => {
    setValidIO(checkInputandOutput(flow))
  },[isBuilt, setValidIO])

  return (
    <>
      <div>
        <BuildTrigger
          open={open}
          flow={flow}
          setIsBuilt={setIsBuilt}
          isBuilt={isBuilt}
        />
        {isBuilt &&
          tabsState[flow.id] &&
          tabsState[flow.id].formKeysData &&
          canOpen && (
            <IOView inputTypes={getInputTypes(flow)} outputTypes={getOutputTypes(flow)}>
              <FormModal
              key={flow.id}
              flow={flow}
              open={open}
              setOpen={setOpen}
            />
            </IOView>
          )}
        <ChatTrigger
          canOpen={canOpen}
          open={open}
          setOpen={setOpen}
          isBuilt={isBuilt}
        />
      </div>
    </>
  );
}
