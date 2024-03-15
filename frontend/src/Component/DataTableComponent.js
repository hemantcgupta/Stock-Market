import React, { Component } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/dist/styles/ag-grid.css';
import 'ag-grid-community/dist/styles/ag-theme-balham.css';
import '../ag-grid-table.scss';
import Spinners from "../spinneranimation";
import cookieStorage from '../Constants/cookie-storage';


class CustomTooltip extends Component {
    getReactContainerClasses() {
        return ['custom-tooltip'];
    }
    render() {
        const tooltipField = this.props.colDef.tooltipField;
        const agGridReact = this.props.agGridReact.gridOptions.rowData;
        if (!agGridReact && agGridReact.length === 0) {
            return;
        }
        const hoveredItem = agGridReact[this.props.rowIndex];
        if (hoveredItem[tooltipField]) {
            return (
                <div className="custom-tooltip">
                    <div class="tooltip-content">{hoveredItem[tooltipField]}</div>
                </div>
            )
        }
        return "";
    }
}

class DataTableComponent extends Component {
    constructor(props) {
        super(props);
        this.state = {
            responsive: false,
            defaultColDef: {
                tooltipComponent: 'customTooltip',
                sortable: true,
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

    isResponsive = () => {
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
        params.api.ensureIndexVisible(this.props.ensureIndexVisible, 'middle')
        let _ = Object.keys(this.props).includes('gridApi') ? this.props.gridApi(params.api) : null
    }

    applyColumnLockDown(columnDefs) {
        return columnDefs.map((c) => {
            if (c.children) {
                c.children = this.applyColumnLockDown(c.children);
            }
            const cellClass = {}
            const alreadyCellBasis = c.cellClassRules;
            if (c.alignLeft && c.underline) {
                cellClass.cellClassRules = {
                    'align-left-underline': '1 == 1',
                    ...alreadyCellBasis,
                };
            } else if (c.alignLeft) {
                cellClass.cellClassRules = {
                    'align-left': '1 == 1',
                    ...alreadyCellBasis,
                };
            } else if (c.underline) {
                cellClass.cellClassRules = {
                    'align-right-underline': '1 == 1',
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
            onSelectionChanged,
            suppressRowClickSelection,
            onCellClicked,
            onCellValueChanged,
            pagination,
            singleRow,
            dashboard,
            agentdashboard,
            channel,
            topMarket,
            loading,
            modal,
            autoHeight,
            frameworkComponents,
            rowClassRules,
            route,
            pos,
            routeDashboard,
            height,
            hideHeader,
            rowSelection,
            getRowStyle
        } = this.props;

        const { responsive } = this.state;

        if (loading) {
            return (
                < Spinners />
            )
        }
        else {
            return (
                <div
                    id='myGrid' className={`ag-theme-balham-dark ag-grid-table ${dashboard ? 'dasboard-height reduce-padding' : topMarket ? 'top-market-height' : autoHeight ? '' : channel ? 'channel-height reduce-padding' : route ? 'reduce-padding route-height' : pos ? 'reduce-padding pos-height' : routeDashboard ? 'route-dashboard' : 'responsive-height'} `}
                    style={{ height: dashboard ? height : agentdashboard ? '20vh' : topMarket ? '26vh' : modal ? '20vh' : autoHeight ? '' : channel ? '45vh' : '28vh' }}
                >
                    <AgGridReact
                        onGridReady={this.onGridReady}
                        onFirstDataRendered={(params) => this.firstDataRendered(params)}
                        columnDefs={this.applyColumnLockDown(columnDefs)}
                        rowData={rowData}
                        suppressRowTransform={true}
                        onSelectionChanged={onSelectionChanged}
                        onCellClicked={onCellClicked}
                        suppressRowClickSelection={suppressRowClickSelection}
                        rowSelection={`${rowSelection ? rowSelection : 'single'}`}
                        domLayout={autoHeight}
                        onCellValueChanged={onCellValueChanged}
                        rowHeight={responsive ? dashboard ? 16 : routeDashboard ? 16 : 28 : 28}
                        // headerHeight={responsive ? dashboard ? 20 : routeDashboard ? 20 : 33 : 33}
                        headerHeight={hideHeader ? 0 : responsive ? dashboard ? 20 : 33 : 33}
                        paginationPageSize={20}
                        pagination={pagination}
                        // Enable when Custom Tolltip Needed
                        // frameworkComponents= {{ customTooltip: CustomTooltip }}
                        rowClassRules={rowClassRules}
                        frameworkComponents={frameworkComponents}
                        enableBrowserTooltips={true}
                        animateRows={true}
                        getRowStyle={getRowStyle}
                    >
                    </AgGridReact>
                </div >
            );
        }

    }
}

export default DataTableComponent;
