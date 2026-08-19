const { app } = window.comfyAPI.app;

app.registerExtension({
  name: "NunchakuKlein.PowerLoraLoader",
  beforeRegisterNodeDef(_nodeType, nodeData) {
    const rows = nodeData.input?.required?.rows;
    if (nodeData.name === "NunchakuKleinPowerLoraLoader" && rows?.[1]?.power_lora_rows) {
      rows[0] = "KLEIN_POWER_LORA_ROWS";
    }
  },
  getCustomWidgets() {
    return {
      KLEIN_POWER_LORA_ROWS(node, inputName, inputData) {
        const loraNames = inputData[1]?.lora_names || [];
        const panel = document.createElement("div");
        panel.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:4px;";
        const rows = [];
        let serializedRows = inputData[1]?.default || "[]";
        let rowsWidget;

        const sync = () => {
          for (const row of rows) {
            row.value = {
              lora_name: row.name.value,
              strength: Number(row.strength.value),
              enabled: row.enabled.checked,
            };
          }
          serializedRows = JSON.stringify(rows.map((row) => row.value));
          rowsWidget.value = serializedRows;
          rowsWidget.callback?.(serializedRows);
          node.onWidgetChanged?.(inputName, serializedRows, undefined, rowsWidget);
          node.graph?.setDirtyCanvas(true, true);
        };

        const render = () => {
          panel.replaceChildren();
          rows.forEach((row, index) => {
            const line = document.createElement("div");
            line.style.cssText = "display:grid;grid-template-columns:auto 1fr 64px repeat(3,28px);gap:3px;";
            row.enabled.title = "Enabled";
            row.enabled.type = "checkbox";
            row.name.style.minWidth = "0";
            row.name.replaceChildren();
            row.strength.type = "number";
            row.strength.min = "-10";
            row.strength.max = "10";
            row.strength.step = "0.05";
            for (const name of loraNames) {
              const option = document.createElement("option");
              option.value = option.textContent = name;
              row.name.append(option);
            }
            row.name.value = row.value.lora_name;
            row.strength.value = row.value.strength;
            row.enabled.checked = row.value.enabled;
            row.enabled.onchange = row.name.onchange = row.strength.onchange = sync;

            const button = (label, title, action) => {
              const element = document.createElement("button");
              element.textContent = label;
              element.title = title;
              element.onclick = action;
              return element;
            };
            line.append(
              row.enabled,
              row.name,
              row.strength,
              button("Up", "Move up", () => {
                if (index > 0) [rows[index - 1], rows[index]] = [rows[index], rows[index - 1]];
                sync(); render();
              }),
              button("Dn", "Move down", () => {
                if (index + 1 < rows.length) [rows[index], rows[index + 1]] = [rows[index + 1], rows[index]];
                sync(); render();
              }),
              button("X", "Remove", () => { rows.splice(index, 1); sync(); render(); }),
            );
            panel.append(line);
          });

          const add = document.createElement("button");
          add.textContent = "+ Add LoRA";
          add.onclick = () => {
            if (!loraNames.length) return;
            rows.push({
              value: { lora_name: loraNames[0], strength: 1, enabled: true },
              enabled: document.createElement("input"),
              name: document.createElement("select"),
              strength: document.createElement("input"),
            });
            render(); sync();
          };
          panel.append(add);
        };

        const loadRows = () => {
          let savedRows = [];
          try { savedRows = JSON.parse(serializedRows || "[]"); } catch (_) {}
          rows.splice(0, rows.length, ...savedRows.map((value) => ({
            value: {
              lora_name: value.lora_name,
              strength: value.strength ?? 1,
              enabled: value.enabled !== false,
            },
            enabled: document.createElement("input"),
            name: document.createElement("select"),
            strength: document.createElement("input"),
          })));
        };

        rowsWidget = node.addDOMWidget(
          inputName,
          "KLEIN_POWER_LORA_ROWS",
          panel,
          {
            getValue: () => serializedRows,
            setValue: (value) => {
              serializedRows = typeof value === "string" ? value : "[]";
              loadRows();
              render();
            },
          },
        );
        loadRows();
        render();
        rowsWidget.options.minNodeSize = [440, 80];
        return { widget: rowsWidget, minWidth: 440, minHeight: 80 };
      },
    };
  },
});
