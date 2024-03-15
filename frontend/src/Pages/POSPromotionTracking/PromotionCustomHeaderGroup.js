import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'
import DownloadCSV from '../../Component/DownloadCSV';

class PromotionCustomHeaderGroup extends Component {
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
        console.log('rahul', displayName)
        const downloadURL = localStorage.getItem('posPromotionDownloadURL')
        return (
            <div>
                {displayName === 'Revenue($)' ?
                    <div className="ag-header-group-cell-label">
                        <div className="customHeaderLabel">{displayName}</div>
                        <DownloadCSV url={downloadURL} name={`Promotion Tracking`} path={`/posPromotion`} page={`Promotion Tracking Page`} />
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

export default PromotionCustomHeaderGroup;