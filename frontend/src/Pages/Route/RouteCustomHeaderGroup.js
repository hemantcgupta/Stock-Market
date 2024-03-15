import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'
import DownloadCSV from '../../Component/DownloadCSV';

class RouteCustomHeaderGroup extends Component {
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
        const {displayName} = this.props;
        const downloadURL = localStorage.getItem('routeDownloadURL')
        return (
            <div>
                {displayName === 'Load Factor' ?
                    <div className="ag-header-group-cell-label">
                        <div className="customHeaderLabel">{displayName}</div>
                        <DownloadCSV url={downloadURL} name={`Route DRILLDOWN`} path={`/route`} page={`Route Page`} />
                    </div> :
                    <div className="ag-header-group-cell-label">
                    <div className="customHeaderLabel">{this.props.displayName}</div>
                    <i class='fa fa-bar-chart-o' style={{ marginLeft: '10px', cursor: 'pointer' }}
                        onClick={() => this.setState({ chartVisible: true })}>
                    </i>
                    </div>}
                <ChartModelDetails
                    chartVisible={this.state.chartVisible}
                    displayName={this.props.displayName}
                    route={true}
                    // datas={this.state.modelRegionDatas}
                    // columns={this.state.modelregioncolumn}
                    closeChartModal={() => this.closeChartModal()}
                />
            </div>
        );
    }

}

export default RouteCustomHeaderGroup;