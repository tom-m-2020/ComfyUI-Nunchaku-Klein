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
        panel.dataset.kleinPowerLoraRows = "";
        panel.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:4px;";
        const rows = [];
        let serializedRows = inputData[1]?.default || "[]";
        let rowsWidget;
        let closeOpenPicker;

        const setPickerValue = (picker, value) => {
          picker.value = value;
          picker.textContent = value || "Select LoRA";
          picker.title = value || "Select LoRA";
        };

        const openLoraPicker = ({ anchor, currentValue, onSelect }) => {
          closeOpenPicker?.();

          const popup = document.createElement("div");
          popup.dataset.kleinLoraPicker = "";
          popup.style.cssText = "position:fixed;z-index:10000;display:flex;flex-direction:column;gap:4px;padding:6px;box-sizing:border-box;background:#202020;color:#f5f5f5;border:1px solid #666;border-radius:4px;box-shadow:0 4px 16px #000a;font:normal 13px/1.4 Arial,sans-serif;";
          const rect = anchor.getBoundingClientRect();
          const width = Math.min(Math.max(rect.width, 360), window.innerWidth - 16);
          popup.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))}px`;
          popup.style.top = `${rect.bottom + 2}px`;
          popup.style.width = `${width}px`;

          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "Search LoRAs";
          search.setAttribute("aria-label", "Search LoRAs");
          search.style.cssText = "box-sizing:border-box;width:100%;height:30px;padding:4px 8px;background:#151515;color:#f5f5f5;border:1px solid #777;border-radius:3px;font:normal 13px/20px Arial,sans-serif;";
          const list = document.createElement("div");
          list.setAttribute("role", "listbox");
          list.setAttribute("aria-label", "LoRAs");
          list.style.cssText = "display:flex;flex-direction:column;max-height:300px;overflow:auto;background:#202020;";
          popup.append(search, list);
          document.body.append(popup);

          let matches = loraNames;
          let activeIndex = Math.max(0, matches.indexOf(currentValue));
          const close = () => {
            document.removeEventListener("pointerdown", onOutside, true);
            popup.remove();
            if (closeOpenPicker === close) closeOpenPicker = undefined;
          };
          const selectActive = () => {
            const name = matches[activeIndex];
            if (name === undefined) return;
            close();
            onSelect(name);
          };
          const updateOptionStyles = () => {
            Array.from(list.children).forEach((option, index) => {
              option.style.background = index === activeIndex
                ? "#315b86"
                : matches[index] === currentValue ? "#383838" : "#202020";
            });
          };
          const renderMatches = () => {
            list.replaceChildren();
            matches.forEach((name, index) => {
              const option = document.createElement("div");
              option.setAttribute("role", "option");
              option.setAttribute("aria-selected", String(name === currentValue));
              option.textContent = name;
              option.title = name;
              option.style.cssText = `box-sizing:border-box;display:flex;align-items:center;flex:0 0 30px;min-height:30px;overflow:hidden;padding:0 10px;color:#f5f5f5;background:${index === activeIndex ? "#315b86" : name === currentValue ? "#383838" : "#202020"};border-left:3px solid ${name === currentValue ? "#7eb6e8" : "transparent"};font:normal 13px/30px Arial,sans-serif;text-align:left;text-overflow:ellipsis;white-space:nowrap;cursor:pointer;`;
              option.onclick = () => {
                activeIndex = index;
                selectActive();
              };
              option.onmouseenter = () => {
                activeIndex = index;
                updateOptionStyles();
              };
              list.append(option);
            });
            list.children[activeIndex]?.scrollIntoView({ block: "nearest" });
          };
          const filter = () => {
            const query = search.value.toLowerCase();
            matches = loraNames.filter((name) => name.toLowerCase().includes(query));
            const selectedIndex = matches.indexOf(currentValue);
            activeIndex = selectedIndex >= 0 ? selectedIndex : 0;
            renderMatches();
          };
          const onOutside = (event) => {
            if (!popup.contains(event.target) && event.target !== anchor) close();
          };
          search.oninput = filter;
          search.onkeydown = (event) => {
            if (event.key === "Escape") {
              event.preventDefault();
              close();
            } else if (event.key === "ArrowDown" && matches.length) {
              event.preventDefault();
              activeIndex = (activeIndex + 1) % matches.length;
              renderMatches();
            } else if (event.key === "ArrowUp" && matches.length) {
              event.preventDefault();
              activeIndex = (activeIndex - 1 + matches.length) % matches.length;
              renderMatches();
            } else if (event.key === "Enter") {
              event.preventDefault();
              selectActive();
            }
          };
          closeOpenPicker = close;
          document.addEventListener("pointerdown", onOutside, true);
          renderMatches();
          search.focus();
        };

        const createPicker = (value) => {
          const picker = document.createElement("button");
          picker.type = "button";
          picker.style.cssText = "min-width:0;overflow:hidden;text-align:left;text-overflow:ellipsis;white-space:nowrap;";
          setPickerValue(picker, value);
          picker.onclick = () => openLoraPicker({
            anchor: picker,
            currentValue: picker.value,
            onSelect: (name) => {
              setPickerValue(picker, name);
              sync();
            },
          });
          return picker;
        };

        const addRow = (loraName) => {
          rows.push({
            value: { lora_name: loraName, strength: 1, enabled: true },
            enabled: document.createElement("input"),
            name: createPicker(loraName),
            strength: document.createElement("input"),
          });
        };

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
          closeOpenPicker?.();
          panel.replaceChildren();
          rows.forEach((row, index) => {
            const line = document.createElement("div");
            line.style.cssText = "display:grid;grid-template-columns:auto 1fr 64px repeat(3,28px);gap:3px;";
            row.enabled.title = "Enabled";
            row.enabled.type = "checkbox";
            row.name.style.minWidth = "0";
            row.strength.type = "number";
            row.strength.min = "-10";
            row.strength.max = "10";
            row.strength.step = "0.05";
            setPickerValue(row.name, row.value.lora_name);
            row.strength.value = row.value.strength;
            row.enabled.checked = row.value.enabled;
            row.enabled.onchange = row.strength.onchange = sync;

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
          add.dataset.kleinAddLora = "";
          add.textContent = "+ Add LoRA";
          add.onclick = () => {
            if (!loraNames.length) return;
            openLoraPicker({
              anchor: add,
              currentValue: null,
              onSelect: (name) => {
                addRow(name);
                render();
                sync();
              },
            });
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
            name: createPicker(value.lora_name),
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
