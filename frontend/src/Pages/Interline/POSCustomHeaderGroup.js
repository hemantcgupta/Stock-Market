import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'
import DownloadCSV from '../../Component/DownloadCSV';

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
        const { displayName } = this.props;
        const downloadURL = localStorage.getItem('posDownloadURL')
        const type = localStorage.getItem('postype')
        console.log(downloadURL, 'latest')
        const groupId = this.props.columnGroup.groupId

        return (
            <div>
                {(type === 'Agency' || type === 'Ancillary') && groupId === '5' ?
                    <div className="customHeaderLabel">{displayName}</div>
                    :
                    displayName === 'AL Market Share(%)' ?
                        <div className="ag-header-group-cell-label">
                            <div className="customHeaderLabel">{displayName}</div>
                            <DownloadCSV url={downloadURL} name={`POS DRILLDOWN`} path={`/pos`} page={`Pos Page`} />
                        </div> :
                        <div className="ag-header-group-cell-label">
                            <div className="customHeaderLabel">{displayName}</div>
                            <i class='fa fa-bar-chart-o' style={{ marginLeft: '10px', cursor: 'pointer' }}
                                onClick={() => this.setState({ chartVisible: true })}>
                            </i>
                        </div>}

                <ChartModelDetails
                    chartVisible={this.state.chartVisible}
                    displayName={displayName}
                    // datas={this.state.modelRegionDatas}
                    // columns={this.state.modelregioncolumn}
                    closeChartModal={() => this.closeChartModal()}
                />
            </div>
        );
    }

}

export default POSCustomHeaderGroup;