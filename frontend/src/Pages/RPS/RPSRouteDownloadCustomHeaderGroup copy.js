import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'
import DownloadCSV from '../../Component/DownloadCSV';

class RPSRouteDownloadCustomHeaderGroup extends Component {
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
        //const flightSelect = localStorage.getItem('FlightSelected')
        //console.log(flightSelect, 'flightSelect')
        const downloadURL = localStorage.getItem('RPSRouteDownloadURL')
        return (
            <div>
                <div className="ag-header-group-cell-label">
                    <div className="customHeaderLabel">{displayName}</div>
                    {/*flightSelect == 'Null' || flightSelect == 'null' || flightSelect == null ? <DownloadCSV url={downloadURL} name={`RPSRoute`} path={`/rpsRos`} page={`RPSRoute Page`} /> : ''*/}
                    <DownloadCSV url={downloadURL} name={`RPSRoute`} path={`/rpsRos`} page={`RPSRoute Page`} />
                </div>

            </div>
        );
    }

}

export default RPSRouteDownloadCustomHeaderGroup;