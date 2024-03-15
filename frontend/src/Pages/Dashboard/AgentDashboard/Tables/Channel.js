import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent';
import TotalRow from '../../../../Component/TotalRow';

const apiServices = new APIServices();

class Channel extends Component {

  constructor(props) {
    super(props);
    this.state = {
      channelData: [],
      channelColumn: [],
      totalData: [],
      totalRO: '',
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    const { gettingYear, gettingMonth, regionId, countryId, cityId, currency } = props;
    this.setState({ loading: true })
    apiServices.getIncrementalRO(gettingYear, gettingMonth, regionId, countryId, cityId, currency).then((result) => {
      this.setState({
        loading: false
      })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        this.setState({ channelData: rowData, channelColumn: columnName, totalData: totalData, totalRO: totalData[0].RO })
      }
    });
  }

  render() {
    return (
      <div >
        <div className="x_title">
          <h2>{`The Incremental Revenue Opportunity is  `}&nbsp;<span className='total-RO'>{`${this.state.totalRO} (${this.props.currency})`}</span></h2>
          <div className="clearfix"></div>
        </div>
        <div className="x_content">
          <DataTableComponent
            rowData={this.state.channelData}
            columnDefs={this.state.channelColumn}
            agentdashboard={true}
            loading={this.state.loading}
          />
          <TotalRow
            rowData={this.state.totalData}
            columnDefs={this.state.channelColumn}
            loading={this.state.loading}
          />
        </div>
      </div>
    )
  }
}

export default Channel;