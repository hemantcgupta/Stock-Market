import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'

class POSCustomHeaderGroup extends Component {
    constructor(props) {
        super(props);
        this.state = {
            chartVisible: false
        }
    }

    closeChartModal() {
        this.setState({ chartVisible: false })
    }

    render() {
        return (
            <div>
                <div className="ag-header-group-cell-label">
                    <div className="customHeaderLabel">{this.props.displayName}</div>
                    <i class='fa fa-bar-chart-o' style={{ marginLeft: '10px', cursor: 'pointer' }}
                        onClick={() => this.setState({ chartVisible: true })}>
                    </i>
                </div>
                <ChartModelDetails
                    chartVisible={this.state.chartVisible}
                    displayName={this.props.displayName}
                    // datas={this.state.modelRegionDatas}
                    // columns={this.state.modelregioncolumn}
                    closeChartModal={() => this.closeChartModal()}
                />
            </div>
        );
    }

}

export default POSCustomHeaderGroup;