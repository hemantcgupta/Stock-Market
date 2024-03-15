import React, { Component } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/dist/styles/ag-grid.css';
import 'ag-grid-community/dist/styles/ag-theme-balham.css';
import '../ag-grid-table.scss';
import Spinners from "../spinneranimation";


class DataTableComponent extends Component {
    constructor(props) {
        super(props);
        this.state = {
            responsive: false,
            defaultColDef: {
                tooltipComponent: 'customTooltip',
                suppressHorizontalScroll: false,
                enableCellChangeFlash: true,
            },
        }
        this.gridApi = null;
    }

    componentWillMount() {
        this.isResponsive()
    }

    componentDidMount() {
        window.onresize = () => {
            this.isResponsive();
        }
    }

    isResponsive() {
        const height = window.innerHeight;
        if (height < 620) {
            this.setState({ responsive: true })
        } else {
            this.setState({ responsive: false })
        }
    }

    firstDataRendered = (params) => {
        params.api.sizeColumnsToFit();
    }

    onGridReady = (params) => {
        if (!params.api) {
            return null;
        }
        this.gridApi = params.api;
        let _ = Object.keys(this.props).includes('gridApi') ? this.props.gridApi(params.api) : null
        // this.gridApi.sizeColumnsToFit();
    }

    applyColumnLockDown(columnDefs) {
        return columnDefs.map((c) => {
            if (c.children) {
                c.children = this.applyColumnLockDown(c.children);
            }
            const cellClass = {}
            const alreadyCellBasis = c.cellClassRules;
            if (c.alignLeft) {
                cellClass.cellClassRules = {
                    'align-left': '1 == 1',
                    ...alreadyCellBasis,
                };
            } else {
                cellClass.cellClassRules = {
                    'align-right': '1 == 1',
                    ...alreadyCellBasis,
                };
            }
            return {
                ...c, ...cellClass, ...{
                    lockPosition: true,
                    cellClass: 'lock-pinned',

                }
            }
        }
        )
    }

    render() {

        setInterval(() => {
            if (this.gridApi) {
                if (this.gridApi.gridPanel.eBodyViewport.clientWidth) {
                    this.gridApi.sizeColumnsToFit();
                }
            }
        }, 1000)

        const {
            rowData,
            columnDefs,
            loading,
            suppressRowClickSelection,
            onSelectionChanged,
            frameworkComponents,
            dashboard,
            reducingPadding,
            route,
            changeBgColor
        } = this.props;

        const { responsive } = this.state;

        if (loading) {
            return (
                ''
            )
        } else {
            return (
                <div
                    id='myGrid' className={`ag-theme-balham-dark ag-grid-table totalRow ${changeBgColor ? "changeBgColor" : ""} ${reducingPadding ? 'reduce-padding' : ''} ${dashboard ? 'totalRow-cell ' : responsive ? 'totalRow-reduce-fontSize' : ''}`}
                    style={{ height: responsive ? dashboard ? '16px' : '30px' : '30px' }}
                >
                    <AgGridReact
                        onGridReady={this.onGridReady}
                        onFirstDataRendered={(params) => this.firstDataRendered(params)}
                        columnDefs={this.applyColumnLockDown(columnDefs)}
                        rowData={rowData}
                        defaultColDef={this.state.defaultColDef}
                        suppressRowTransform={true}
                        onSelectionChanged={onSelectionChanged}
                        suppressRowClickSelection={suppressRowClickSelection}
                        rowSelection={`single`}
                        headerHeight="0"
                        rowStyle={{ fontWeight: 'bold' }}
                        enableBrowserTooltips={true}
                        frameworkComponents={frameworkComponents}
                        rowHeight={responsive ? dashboard ? 16 : 28 : 28}
                    >
                    </AgGridReact>
                </div >
            );
        }
    }
}

export default DataTableComponent;
